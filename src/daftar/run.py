"""``track()`` and ``@tracked`` -- the two things a user actually types.

The entire adoption argument rests on this file being unobtrusive. If wrapping
an existing simulation costs more than one line and one indent level, people
will not do it, and everything downstream is irrelevant. So: no new DSL, no
config file, no daemon, no account. A context manager and a decorator.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import capture
from .manifest import (
    NS_CODE, NS_COST, NS_ENV, NS_INPUT, NS_META, NS_OUTPUT, NS_PARAM,
    NS_RESULT, NS_SEED, Manifest,
)
from .store import RunStore


def _new_run_id() -> str:
    """Short, sortable-ish, and collision-free enough for a single lab."""
    return "r-" + uuid.uuid4().hex[:8]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _package_dir() -> str:
    return str(Path(__file__).resolve().parent)


def _caller_entrypoint() -> str:
    """Best-effort ``path/to/file.py::function`` for the code being tracked.

    Walks outward until it leaves daftar and the stdlib plumbing that sits
    between us and the user. An earlier version counted a fixed number of
    frames, which worked when called directly and resolved to
    ``_pytest/python.py`` under pytest -- the frame depth depends on who is
    calling, so counting frames was never going to hold.
    """
    pkg = _package_dir()
    skip_files = {"contextlib.py", "functools.py"}
    try:
        for frame in inspect.stack()[1:]:
            filename = frame.filename
            resolved = str(Path(filename).resolve())
            if resolved.startswith(pkg):
                continue
            if Path(filename).name in skip_files:
                continue
            if "importlib" in resolved or resolved.startswith("<"):
                continue
            path = Path(filename)
            try:
                path = path.relative_to(Path.cwd())
            except ValueError:
                pass
            fn = frame.function
            return f"{path.as_posix()}::{fn}" if fn != "<module>" else path.as_posix()
    except Exception:  # pragma: no cover - introspection is never critical
        pass
    return "unknown"


class Run:
    """A single tracked execution. Yielded by :func:`track`."""

    def __init__(
        self,
        label: str = "",
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        notes: str = "",
        store: RunStore | None = None,
        tags: list[str] | None = None,
        entrypoint: str | None = None,
        capture_env: bool = True,
        capture_git: bool = True,
    ):
        self.run_id = _new_run_id()
        self.manifest = Manifest(run_id=self.run_id)
        self.store = store or RunStore()
        self.label = label
        self.status = "running"
        self._t0: float | None = None
        self._seed_value = capture.resolve_seed(seed)
        self._seed_requested = seed is not None
        self._capture_env = capture_env
        self._capture_git = capture_git
        self._entrypoint = entrypoint
        self._notes = notes
        self._tags = tags or []
        self._initial_params = dict(params or {})

    def _git_excludes(self) -> list[str]:
        """Paths git should ignore when deciding whether the tree is dirty."""
        excludes = {self.store.dir.name}
        try:
            excludes.add(self.store.dir.relative_to(Path.cwd()).as_posix())
        except ValueError:
            pass  # store lives outside the tree; the bare name is enough
        return sorted(excludes)

    # -- lifecycle --------------------------------------------------------

    def _begin(self) -> "Run":
        m = self.manifest
        self._t0 = time.perf_counter()

        m.set(f"{NS_META}.run_id", self.run_id)
        m.set(f"{NS_META}.label", self.label)
        m.set(f"{NS_META}.started_at", _utcnow())
        m.set(f"{NS_META}.status", "running")
        m.set(f"{NS_META}.daftar_schema", "1")
        if self._notes:
            m.set(f"{NS_META}.notes", self._notes)
        if self._tags:
            m.set(f"{NS_META}.tags", sorted(self._tags))

        m.set(f"{NS_CODE}.entrypoint", self._entrypoint or _caller_entrypoint())
        m.set(f"{NS_CODE}.argv", " ".join(sys.argv))
        if self._capture_git:
            # Exclude our own store, so using daftar does not make the repo
            # look dirty to daftar.
            m.set_many(NS_CODE, capture.git_state(exclude=self._git_excludes()))

        # Seeds are applied, not merely noted. See capture.apply_seeds.
        applied = capture.apply_seeds(self._seed_value)
        m.set_many(NS_SEED, applied)
        m.set(f"{NS_SEED}.was_explicit", self._seed_requested)

        self.log_params(self._initial_params)
        return self

    def _end(self, status: str, error: BaseException | None = None) -> None:
        m = self.manifest
        self.status = status
        m.set(f"{NS_META}.status", status)
        m.set(f"{NS_META}.finished_at", _utcnow())
        if self._t0 is not None:
            m.set(f"{NS_COST}.wall_clock_s", round(time.perf_counter() - self._t0, 3))
        m.set_many(NS_COST, capture.platform_details())

        # Environment last: by now every library the run needed has been
        # imported, so this is the set that actually mattered.
        if self._capture_env:
            m.set_many(NS_ENV, capture.environment())

        if error is not None:
            m.set(f"{NS_META}.error_type", type(error).__name__)
            m.set(f"{NS_META}.error", str(error)[:500])
            tb = "".join(traceback.format_exception_only(type(error), error)).strip()
            m.set(f"{NS_META}.error_summary", tb[:500])

        self.store.save(m)

    # -- the recording API ------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Record something you chose."""
        self.manifest.set(f"{NS_PARAM}.{key}", value)

    def log_params(self, params: dict[str, Any], prefix: str = "") -> None:
        for k, v in (params or {}).items():
            name = f"{prefix}{k}"
            if isinstance(v, dict):
                self.log_params(v, prefix=f"{name}.")
            else:
                self.log_param(name, v)

    def log_result(self, key: str, value: Any) -> None:
        """Record a scalar outcome worth comparing across runs.

        Keep these small and few. A result field is something you would put in a
        table in a paper; dumping a 10,000-element array here makes every diff
        useless, which is the failure mode that kills these tools.
        """
        self.manifest.set(f"{NS_RESULT}.{key}", value)

    def log_results(self, results: dict[str, Any]) -> None:
        for k, v in (results or {}).items():
            self.log_result(k, v)

    def add_input(self, path: str | os.PathLike, name: str | None = None) -> str:
        """Hash an input file or directory and record it."""
        p = Path(path)
        digest, size = capture.hash_path(p)
        key = name or p.name
        self.manifest.set(f"{NS_INPUT}.{key}.sha256", capture.short_hash(digest))
        self.manifest.set(f"{NS_INPUT}.{key}.bytes", size)
        self.manifest.set(f"{NS_INPUT}.{key}.path", p.as_posix())
        return digest

    def add_output(self, path: str | os.PathLike, name: str | None = None) -> str | None:
        """Hash a produced file. Missing files are recorded as missing, not fatal.

        A run that produced results but failed to write one figure should still
        yield a manifest. Losing the whole record over a missing PNG is the kind
        of brittleness that gets a tool removed.
        """
        p = Path(path)
        key = name or p.name
        self.manifest.set(f"{NS_OUTPUT}.{key}.path", p.as_posix())
        if not p.exists():
            self.manifest.set(f"{NS_OUTPUT}.{key}.sha256", "missing")
            return None
        digest, size = capture.hash_path(p)
        self.manifest.set(f"{NS_OUTPUT}.{key}.sha256", capture.short_hash(digest))
        self.manifest.set(f"{NS_OUTPUT}.{key}.bytes", size)
        return digest

    def note(self, text: str) -> None:
        self.manifest.set(f"{NS_META}.notes", text)

    @property
    def seed(self) -> int:
        """The seed actually in use. Derive JAX keys from this."""
        return self._seed_value

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Run {self.run_id} {self.label!r} {self.status}>"


class _TrackContext:
    def __init__(self, run: Run):
        self._run = run

    def __enter__(self) -> Run:
        return self._run._begin()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self._run._end("completed")
        elif isinstance(exc, KeyboardInterrupt):
            self._run._end("interrupted", exc)
        else:
            self._run._end("failed", exc)
        return False  # never swallow the exception


def track(
    label: str = "",
    *,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    notes: str = "",
    tags: list[str] | None = None,
    store: RunStore | None = None,
    entrypoint: str | None = None,
    capture_env: bool = True,
    capture_git: bool = True,
) -> _TrackContext:
    """Track a block of code.

    ::

        with daftar.track("celegans", params={"dt": 0.025}, seed=42) as run:
            run.add_input("data/connectome.csv")
            v = simulate(dt=0.025)
            run.log_result("mean_rate_hz", float(v.mean()))

    Failed runs are recorded too, with ``meta.status = failed``. A crashed run
    that took four hours is exactly the one you will want to look at later.
    """
    return _TrackContext(Run(
        label=label, params=params, seed=seed, notes=notes, tags=tags,
        store=store, entrypoint=entrypoint,
        capture_env=capture_env, capture_git=capture_git,
    ))


def tracked(
    _fn: Callable | None = None,
    *,
    label: str | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
    store: RunStore | None = None,
    capture_env: bool = True,
    capture_git: bool = True,
) -> Callable:
    """Decorator form. Arguments become params; a returned dict becomes results.

    ::

        @daftar.tracked(seed=42)
        def simulate(dt=0.025, solver="bwd_euler"):
            ...
            return {"mean_rate_hz": 4.81}

    Defaults are bound and recorded too, so a parameter you never passed still
    appears in the manifest -- which is the one you will change six weeks later
    and forget about.
    """
    def decorate(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = {
                k: v for k, v in bound.arguments.items()
                if not k.startswith("_")
            }
            entry = f"{Path(inspect.getfile(fn)).name}::{fn.__name__}"
            ctx = _TrackContext(Run(
                label=label or fn.__name__, params=params, seed=seed, tags=tags,
                store=store, entrypoint=entry,
                capture_env=capture_env, capture_git=capture_git,
            ))
            with ctx as run:
                # Let the function opt in to the run object if it wants one.
                if "run" in sig.parameters and "run" not in kwargs:
                    kwargs["run"] = run
                out = fn(*args, **kwargs)
                if isinstance(out, dict):
                    run.log_results({
                        k: v for k, v in out.items()
                        if isinstance(v, (int, float, str, bool))
                    })
                wrapper.last_run_id = run.run_id
                return out

        wrapper.last_run_id = None
        return wrapper

    return decorate(_fn) if callable(_fn) else decorate
