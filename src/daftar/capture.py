"""Capturing the things nobody writes down.

The most important function here is :func:`apply_seeds`, and it is worth saying
why. A provenance tool that only *records* the seed is close to useless: if the
code seeded itself from the clock, recording that fact tells you the run is
irreproducible but does nothing to fix it. So daftar sets the seeds itself,
across every RNG it can find, and then records what it set. Recording follows
from controlling, not the other way round.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import random
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Packages worth recording without being asked. Anything actually imported by
# the process is also picked up by ``imported_package_versions``; this list just
# guarantees the usual suspects appear even if imported lazily later.
_ALWAYS_RECORD = (
    "numpy", "scipy", "pandas", "jax", "jaxlib", "jaxley", "torch",
    "equinox", "brian2", "neuron", "cpm-toolbox", "cpm", "dm-meltingpot",
    "dmlab2d", "gdm-concordia", "matplotlib", "scikit-learn",
)

_HASH_CHUNK = 1 << 20  # 1 MiB


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------

def _git(*args: str, cwd: str | os.PathLike | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state(
    cwd: str | os.PathLike | None = None,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Commit, dirtiness, branch, and a hash of uncommitted changes.

    ``code.dirty_diff_sha256`` matters more than it looks. "Dirty" alone tells
    you the working tree had edits but not whether *the same* edits; hashing the
    diff means two dirty runs can still be shown to be identical, which is the
    common case during a debugging session.

    ``exclude`` holds paths to leave out of the dirtiness calculation -- in
    practice, daftar's own store. Without it the tool reports its own
    bookkeeping as a change to your experiment: the first run is clean, and
    every run after it is dirty because ``.daftar/runs/`` now exists. A
    provenance tool that perturbs the thing it measures is worse than none.
    """
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return {"vcs": "none"}

    state: dict[str, Any] = {
        "vcs": "git",
        "commit": commit,
        "commit_short": commit[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd) or "unknown",
    }

    # Git pathspec magic; supported since 1.9 and a no-op when nothing matches.
    pathspec = ["--", "."] + [f":(exclude){p}" for p in exclude] if exclude else []

    diff = _git("diff", "HEAD", *pathspec, cwd=cwd)
    untracked = _git(
        "ls-files", "--others", "--exclude-standard", *pathspec, cwd=cwd
    )
    dirty = bool(diff) or bool(untracked)
    state["dirty"] = dirty
    if dirty:
        h = hashlib.sha256()
        h.update((diff or "").encode())
        h.update(b"\0")
        h.update((untracked or "").encode())
        state["dirty_diff_sha256"] = h.hexdigest()[:16]
    remote = _git("config", "--get", "remote.origin.url", cwd=cwd)
    if remote:
        state["remote"] = remote
    return state


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except Exception:
        return None


def imported_package_versions() -> dict[str, str]:
    """Versions of top-level packages currently imported, plus the usual suspects.

    Recording the whole environment is tempting and wrong: a 300-package
    ``pip freeze`` buries the one line that matters under noise, and every
    unrelated upgrade produces a spurious diff. We record what the process
    actually touched.
    """
    names: set[str] = set(_ALWAYS_RECORD)
    for mod in list(sys.modules):
        if "." in mod or mod.startswith("_"):
            continue
        names.add(mod)

    found: dict[str, str] = {}
    for name in names:
        v = _version(name) or _version(name.replace("_", "-"))
        if v:
            found[name.replace("-", "_")] = v
    return found


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
    }
    env.update(imported_package_versions())
    return env


def platform_details() -> dict[str, Any]:
    details: dict[str, Any] = {"hostname": platform.node()}
    try:
        details["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    return details


# --------------------------------------------------------------------------
# file hashing
# --------------------------------------------------------------------------

def hash_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: str | os.PathLike) -> tuple[str, int]:
    """Hash a file, or a directory as a whole. Returns ``(sha256, n_bytes)``.

    Directory hashing walks in sorted order and mixes in relative paths, so a
    renamed file changes the hash. Without that, moving ``a.csv`` to ``b.csv``
    would look like no change at all.
    """
    p = Path(path)
    if p.is_file():
        return hash_file(p), p.stat().st_size

    if p.is_dir():
        h = hashlib.sha256()
        total = 0
        for child in sorted(p.rglob("*")):
            if not child.is_file():
                continue
            rel = child.relative_to(p).as_posix()
            h.update(rel.encode())
            h.update(b"\0")
            h.update(hash_file(child).encode())
            h.update(b"\n")
            total += child.stat().st_size
        return h.hexdigest(), total

    raise FileNotFoundError(path)


def short_hash(full: str, n: int = 12) -> str:
    return full[:n]


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def apply_seeds(seed: int) -> dict[str, Any]:
    """Seed every RNG we can reach. Returns what was actually set.

    Only libraries already imported are touched -- importing torch to seed it
    would add ten seconds to a run that never uses it. The returned dict is
    recorded verbatim, so a manifest tells you not just the seed but which
    generators it reached.
    """
    applied: dict[str, Any] = {"value": seed, "python_random": True}
    random.seed(seed)

    # Affects hash randomisation only if set before interpreter start, but
    # recording it is still useful for explaining set-iteration differences.
    applied["pythonhashseed_env"] = os.environ.get("PYTHONHASHSEED", "unset")

    np = sys.modules.get("numpy")
    if np is not None:
        try:
            np.random.seed(seed)
            applied["numpy_legacy"] = True
        except Exception:  # pragma: no cover
            applied["numpy_legacy"] = False

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.manual_seed(seed)
            applied["torch"] = True
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                applied["torch_cuda"] = True
        except Exception:  # pragma: no cover
            applied["torch"] = False

    # JAX is functional: there is no global RNG to set. The honest thing is to
    # record the root key the user should be deriving from, and say so.
    jax = sys.modules.get("jax")
    if jax is not None:
        applied["jax_root_key"] = f"jax.random.PRNGKey({seed})"

    return applied


def resolve_seed(seed: int | None) -> int:
    """Pick a seed if the caller did not.

    Choosing one and recording it beats letting the library seed itself from
    entropy we never see. An accidental seed that is written down is
    reproducible; a deliberate one that isn't, is not.
    """
    if seed is not None:
        return int(seed)
    return int.from_bytes(os.urandom(4), "big")
