"""What an adapter is, and what makes one worth writing.

The core package can already track any Python function. An adapter earns its
existence only by knowing something domain-specific that a generic tracker
cannot infer:

* which arguments are *scientifically* meaningful parameters, as opposed to
  file paths and verbosity flags;
* where the framework hides state that silently changes results -- a solver
  tolerance, an estimator choice, a substrate revision;
* what counts as a comparable scalar result in that field.

An adapter that just calls ``log_params(kwargs)`` is not worth the import. The
test is whether it records something the researcher would have forgotten.

Adapters must never import their framework at module import time. Someone with
Jaxley installed but not MeltingPot has to be able to ``import daftar``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..run import Run


@runtime_checkable
class Adapter(Protocol):
    """Structural protocol. Adapters are modules, not classes."""

    name: str

    def is_available(self) -> bool:
        """True if the target framework is importable."""

    def describe(self, obj: Any, run: Run, prefix: str = "") -> None:
        """Record everything provenance-relevant about ``obj`` into ``run``."""


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, name: str, module: Any) -> None:
        self._adapters[name] = module

    def get(self, name: str) -> Any:
        if name not in self._adapters:
            raise KeyError(
                f"no adapter named {name!r}; available: "
                f"{', '.join(sorted(self._adapters)) or 'none'}"
            )
        return self._adapters[name]

    def available(self) -> list[str]:
        out = []
        for name, mod in self._adapters.items():
            try:
                if mod.is_available():
                    out.append(name)
            except Exception:
                continue
        return sorted(out)

    def all(self) -> list[str]:
        return sorted(self._adapters)


def safe(fn, default=None):
    """Run a probe that may fail against an unfamiliar framework version.

    Adapters read private-ish attributes of fast-moving research code. A
    provenance tool that crashes a four-hour simulation because a framework
    renamed an attribute has done far more harm than the missing field is worth.
    Every probe is best-effort; a missing field is recorded as missing.
    """
    try:
        return fn()
    except Exception:
        return default


#: An adapter's target framework is in one of three states, and conflating the
#: last two is how a broken environment gets mistaken for an absent one.
AVAILABLE = "available"
MISSING = "missing"
BROKEN = "broken"


def probe_import(module_name: str) -> tuple[str, str]:
    """Try to import a framework. Returns ``(status, reason)``.

    ``ImportError`` means the package is not installed, which is ordinary and
    uninteresting. **Any other exception means it is installed and broken** --
    an incompatible NumPy, a missing shared library, a version clash with
    another package. Those two cases need different responses from the user and
    must not look identical.

    This distinction exists because a bare ``try: import x except: False``
    reports a broken install as an absent one. The test suite then skips
    quietly, the adapter is never exercised, and nothing anywhere says why.
    """
    try:
        __import__(module_name)
        return AVAILABLE, ""
    except ImportError as exc:
        # A broken C extension also raises ImportError, so check the message.
        text = str(exc)
        if "No module named" in text:
            return MISSING, "not installed"
        return BROKEN, f"{type(exc).__name__}: {text}"
    except Exception as exc:
        return BROKEN, f"{type(exc).__name__}: {exc}"


def record_optional(run: Run, key: str, fn, *, kind: str = "param") -> None:
    """Record ``fn()`` under ``key``, or note that it could not be read."""
    value = safe(fn, default="<unavailable>")
    if kind == "param":
        run.log_param(key, value)
    elif kind == "result":
        run.log_result(key, value)
    else:
        run.manifest.set(f"{kind}.{key}", value)
