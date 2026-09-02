"""Parameter sweeps, replay, and self-contained export."""

from __future__ import annotations

import itertools
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .diff import compare_many
from .manifest import NS_PARAM, Manifest
from .run import Run, _TrackContext
from .store import RunStore


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

@dataclass
class SweepResult:
    run_ids: list[str]
    label: str
    failed: list[tuple[str, str]]

    def table(self, store: RunStore | None = None) -> list[dict[str, str]]:
        """One row per run, with only the columns that actually varied."""
        store = store or RunStore()
        manifests = [store.load(r) for r in self.run_ids]
        varying = compare_many(manifests)
        cols = sorted(k for k in varying if k.startswith(f"{NS_PARAM}.")
                      or k.startswith("result."))
        rows = []
        for m in manifests:
            row = {"run_id": m.run_id, "status": m.get("meta.status", "")}
            for c in cols:
                row[c] = m.fields.get(c, "")
            rows.append(row)
        return rows


def grid(**axes: list[Any]) -> Iterator[dict[str, Any]]:
    """Cartesian product of named axes, in declaration order.

    ::

        for combo in grid(dt=[0.025, 0.01], solver=["bwd_euler", "crank_nicolson"]):
            ...
    """
    names = list(axes)
    for values in itertools.product(*(axes[n] for n in names)):
        yield dict(zip(names, values))


def sweep(
    fn: Callable[..., Any],
    *,
    label: str = "sweep",
    seed: int | None = None,
    store: RunStore | None = None,
    continue_on_error: bool = True,
    **axes: list[Any],
) -> SweepResult:
    """Run ``fn(**params)`` once per grid point, tracking each separately.

    Each combination is a separate run with its own manifest, rather than one
    run with a nested table. That means the sweep and a run you did by hand last
    Tuesday are the same kind of object, and ``diff`` works across them.

    ``continue_on_error`` defaults to True: a sweep that dies on point 3 of 40
    and discards the first two results is worse than useless.
    """
    store = store or RunStore()
    run_ids: list[str] = []
    failed: list[tuple[str, str]] = []

    for combo in grid(**axes):
        ctx = _TrackContext(Run(
            label=label, params=combo, seed=seed, store=store,
            tags=["sweep"],
        ))
        try:
            with ctx as run:
                out = fn(**combo)
                if isinstance(out, dict):
                    run.log_results({
                        k: v for k, v in out.items()
                        if isinstance(v, (int, float, str, bool))
                    })
                run_ids.append(run.run_id)
        except Exception as exc:
            run_ids.append(ctx._run.run_id)
            failed.append((ctx._run.run_id, f"{type(exc).__name__}: {exc}"))
            if not continue_on_error:
                raise

    return SweepResult(run_ids=run_ids, label=label, failed=failed)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

@dataclass
class ReplayPlan:
    """What it would take to reproduce a run, and what stands in the way.

    Deliberately *not* an executor. Re-running arbitrary recorded code would
    mean this package executes whatever a manifest tells it to, which is both a
    security problem and a lie -- we cannot restore someone's CUDA driver. What
    we can do honestly is state the target state, check the current state
    against it, and list every discrepancy.
    """
    run_id: str
    entrypoint: str
    commit: str
    dirty: bool
    seed: str
    params: dict[str, str]
    inputs: dict[str, str]
    env: dict[str, str]          # package -> version
    sources: dict[str, str]      # package -> install origin (PEP 610), if any
    blockers: list[str]
    warnings: list[str]

    @property
    def reproducible(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        lines = [f"Replay plan for {self.run_id}", ""]
        lines.append(f"  entrypoint   {self.entrypoint}")
        lines.append(f"  commit       {self.commit}{'  (DIRTY)' if self.dirty else ''}")
        lines.append(f"  seed         {self.seed}")
        if self.params:
            lines.append("")
            lines.append("  parameters")
            w = max(len(k) for k in self.params)
            for k, v in sorted(self.params.items()):
                lines.append(f"    {k.ljust(w)}  {v}")
        if self.inputs:
            lines.append("")
            lines.append("  inputs")
            for k, v in sorted(self.inputs.items()):
                lines.append(f"    {k}  {v}")
        lines.append("")
        if self.blockers:
            lines.append("  BLOCKERS -- this run cannot be reproduced as recorded:")
            for b in self.blockers:
                lines.append(f"    - {b}")
        else:
            lines.append("  No blockers. The recorded state is recoverable.")
        if self.warnings:
            lines.append("")
            lines.append("  warnings")
            for w_ in self.warnings:
                lines.append(f"    - {w_}")
        lines.append("")
        lines.append("  To reproduce:")
        if self.commit and self.commit != "null":
            lines.append(f"    git checkout {self.commit}")
        pkgs = sorted(self.env.items())
        total = len(pkgs)
        if total > 12:
            lines.append(
                f"    # {total} packages recorded; see env.* in the manifest"
            )
            pkgs = pkgs[:12]

        # A package installed from a VCS or a local path cannot be pinned by
        # version: the version string does not identify the code. Emit the
        # origin instead, and separate out the ones pip cannot fetch at all.
        specs: list[str] = []
        manual: list[tuple[str, str]] = []
        for name, version in pkgs:
            origin = self.sources.get(name)
            if origin is None:
                specs.append(f"{name}=={version}")
            elif origin.startswith(("git+", "hg+", "svn+", "bzr+")):
                url = origin.replace("#", "@", 1) if "#" in origin else origin
                specs.append(f'"{name} @ {url}"')
            else:
                manual.append((name, origin))

        if specs:
            lines.append("    pip install " + " ".join(specs))
        for name, origin in manual:
            lines.append(
                f"    # {name}: installed from {origin} -- not fetchable by pip;"
                f" obtain this source separately"
            )
        lines.append(f"    # then run {self.entrypoint} with the parameters above")
        return "\n".join(lines)


def plan_replay(manifest: Manifest, *, check_current: bool = True) -> ReplayPlan:
    from . import capture

    blockers: list[str] = []
    warnings: list[str] = []

    commit = manifest.get("code.commit", "") or ""
    dirty = manifest.get("code.dirty", "false") == "true"
    if dirty:
        blockers.append(
            "working tree was dirty; uncommitted changes were not preserved "
            f"(diff hash {manifest.get('code.dirty_diff_sha256', 'unknown')})"
        )
    if not commit:
        warnings.append("run was not made inside a git repository")

    if manifest.get("seed.was_explicit") == "false":
        warnings.append(
            "seed was auto-generated rather than chosen; it was recorded and "
            "can be reused, but the original code did not request it"
        )

    params = {k.split(".", 1)[1]: v for k, v in manifest.namespace(NS_PARAM).items()}
    inputs = {
        k.rsplit(".", 1)[0].split(".", 1)[1]: v
        for k, v in manifest.namespace("input").items() if k.endswith(".sha256")
    }
    env_ns = manifest.namespace("env")
    env = {
        k.split(".", 1)[1]: v for k, v in env_ns.items()
        if not k.endswith(".source")
    }
    sources = {
        k.split(".", 1)[1].rsplit(".source", 1)[0]: v
        for k, v in env_ns.items() if k.endswith(".source")
    }
    # Editable and local-path installs are the honest hard case: nobody else can
    # fetch "file:///Users/you/Downloads/thing", and the version string says
    # nothing about what code was actually there.
    for pkg, origin in sorted(sources.items()):
        if origin.startswith(("editable:", "local:")):
            warnings.append(
                f"{pkg} was installed from a local path ({origin}); its version "
                f"string does not identify the code and no one else can fetch it"
            )

    if check_current:
        # Verify recorded input files still hash the same.
        for k, v in manifest.namespace("input").items():
            if not k.endswith(".path"):
                continue
            name = k.rsplit(".", 1)[0]
            recorded = manifest.get(f"{name}.sha256")
            p = Path(v)
            if not p.exists():
                blockers.append(f"input file missing: {v}")
            else:
                try:
                    digest, _ = capture.hash_path(p)
                    if capture.short_hash(digest) != recorded:
                        blockers.append(
                            f"input file changed since the run: {v} "
                            f"(was {recorded}, now {capture.short_hash(digest)})"
                        )
                except OSError as exc:
                    warnings.append(f"could not hash {v}: {exc}")

        current = capture.environment()
        for pkg, version in env.items():
            if pkg in ("python", "os", "os_release", "machine", "python_impl"):
                if str(current.get(pkg, "")) != version:
                    warnings.append(
                        f"{pkg} differs: recorded {version}, current "
                        f"{current.get(pkg, 'unknown')}"
                    )
            elif pkg in current and str(current[pkg]) != version:
                warnings.append(
                    f"package {pkg} differs: recorded {version}, current {current[pkg]}"
                )

    return ReplayPlan(
        run_id=manifest.run_id,
        entrypoint=manifest.get("code.entrypoint", "unknown"),
        commit=commit,
        dirty=dirty,
        seed=manifest.get("seed.value", "unknown"),
        params=params,
        inputs=inputs,
        env={k: v for k, v in env.items()
             if k not in ("os", "os_release", "machine", "python_impl", "python")},
        sources=sources,
        blockers=blockers,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_bundle(
    manifest: Manifest,
    out_path: str | Path,
    *,
    store: RunStore | None = None,
    include_inputs: bool = True,
    include_outputs: bool = True,
    max_bytes: int = 512 * 1024 * 1024,
) -> Path:
    """Write a self-contained archive: manifest, README, and referenced files.

    The README is generated in plain English and placed at the archive root, so
    the recipient learns what they are looking at without installing anything.
    A bundle that requires our tool to be understood defeats its own purpose.
    """
    store = store or RunStore()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plan = plan_replay(manifest, check_current=False)
    readme = _bundle_readme(manifest, plan)

    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", readme)
        zf.writestr("manifest.json", manifest.to_json())
        zf.writestr("fields.tsv", _fields_tsv(manifest))

        for ns, flag, folder in (
            ("input", include_inputs, "inputs"),
            ("output", include_outputs, "outputs"),
        ):
            if not flag:
                continue
            for key, value in manifest.namespace(ns).items():
                if not key.endswith(".path"):
                    continue
                p = Path(value)
                if not p.is_file():
                    continue
                size = p.stat().st_size
                if total + size > max_bytes:
                    zf.writestr(
                        f"{folder}/OMITTED-{p.name}.txt",
                        f"{p} omitted: bundle size limit of {max_bytes} bytes reached.\n"
                        f"Recorded sha256 prefix: "
                        f"{manifest.get(key.rsplit('.', 1)[0] + '.sha256', 'unknown')}\n",
                    )
                    continue
                zf.write(p, f"{folder}/{p.name}")
                total += size

    return out


def _fields_tsv(m: Manifest) -> str:
    rows = ["key\tvalue"]
    rows += [f"{k}\t{m.fields[k]}" for k in m.ordered_keys]
    return "\n".join(rows) + "\n"


def _bundle_readme(m: Manifest, plan: ReplayPlan) -> str:
    lines = [
        f"# Run {m.run_id}",
        "",
        f"**Label:** {m.label or '(none)'}  ",
        f"**Started:** {m.started_at}  ",
        f"**Status:** {m.get('meta.status', 'unknown')}  ",
        f"**Duration:** {m.get('cost.wall_clock_s', 'unknown')} s",
        "",
        "This archive was produced by daftar. It records what generated a",
        "computational result: the code version, parameters, random seeds, input",
        "file hashes, and software environment.",
        "",
        "You do not need daftar to read it. `manifest.json` and `fields.tsv`",
        "are plain text and contain everything.",
        "",
        "## What produced this",
        "",
        f"- Entry point: `{plan.entrypoint}`",
        f"- Git commit: `{plan.commit or 'not in a repository'}`"
        + ("  **(uncommitted changes were present)**" if plan.dirty else ""),
        f"- Random seed: `{plan.seed}`",
        "",
    ]
    if plan.params:
        lines += ["## Parameters", ""]
        lines += [f"- `{k}` = `{v}`" for k, v in sorted(plan.params.items())]
        lines.append("")
    results = m.namespace("result")
    if results:
        lines += ["## Results", ""]
        lines += [f"- `{k.split('.', 1)[1]}` = `{v}`" for k, v in sorted(results.items())]
        lines.append("")
    if plan.inputs:
        lines += ["## Inputs (sha256 prefix)", ""]
        lines += [f"- `{k}`: `{v}`" for k, v in sorted(plan.inputs.items())]
        lines.append("")
    if plan.env:
        lines += ["## Environment", ""]
        for k, v in sorted(plan.env.items()):
            origin = plan.sources.get(k)
            lines.append(f"- {k} {v}" + (f"  (from {origin})" if origin else ""))
        lines.append("")
    if plan.blockers:
        lines += ["## Known obstacles to exact reproduction", ""]
        lines += [f"- {b}" for b in plan.blockers]
        lines.append("")
    return "\n".join(lines)


def load_bundle(path: str | Path) -> Manifest:
    with zipfile.ZipFile(path) as zf:
        return Manifest.from_dict(json.loads(zf.read("manifest.json")))
