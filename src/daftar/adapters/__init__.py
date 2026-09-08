"""Framework adapters.

Import is lazy on purpose: someone with Jaxley but not MeltingPot installed must
still be able to ``import daftar``. Nothing here imports a target framework
at module load.
"""

from __future__ import annotations

from . import (
    brian2_adapter, cpm_adapter, jaxley_adapter, meltingpot_adapter, mne_adapter,
    gdsfactory_adapter, netpyne_adapter, nilearn_adapter, sbi_adapter,
)
from .base import (
    AVAILABLE, BROKEN, MISSING, Adapter, AdapterRegistry, probe_import,
    record_optional, safe,
)

registry = AdapterRegistry()
registry.register("jaxley", jaxley_adapter)
registry.register("cpm", cpm_adapter)
registry.register("meltingpot", meltingpot_adapter)
registry.register("brian2", brian2_adapter)
registry.register("mne", mne_adapter)
registry.register("sbi", sbi_adapter)
registry.register("nilearn", nilearn_adapter)
registry.register("gdsfactory", gdsfactory_adapter)
registry.register("netpyne", netpyne_adapter)

jaxley = jaxley_adapter
cpm = cpm_adapter
meltingpot = meltingpot_adapter
brian2 = brian2_adapter
mne = mne_adapter
sbi = sbi_adapter
nilearn = nilearn_adapter
gdsfactory = gdsfactory_adapter
netpyne = netpyne_adapter


def available() -> list[str]:
    """Adapters whose target framework is importable right now."""
    return registry.available()


def status() -> dict[str, tuple[str, str]]:
    """``{adapter: (status, reason)}`` for every registered adapter.

    Distinguishes an absent framework from a broken one. A framework that is
    installed but fails to import -- an incompatible NumPy, a missing shared
    library -- is a situation the user can fix, and it must not be reported the
    same way as one they simply have not installed.
    """
    out: dict[str, tuple[str, str]] = {}
    for name in registry.all():
        mod = registry.get(name)
        try:
            out[name] = mod.availability()
        except Exception as exc:  # pragma: no cover - defensive
            out[name] = (BROKEN, f"{type(exc).__name__}: {exc}")
    return out


def get(name: str):
    return registry.get(name)


__all__ = [
    "registry", "available", "get",
    "jaxley", "cpm", "meltingpot", "brian2", "mne", "sbi", "nilearn", "gdsfactory", "netpyne",
    "jaxley_adapter", "cpm_adapter", "meltingpot_adapter", "brian2_adapter",
    "mne_adapter", "sbi_adapter", "nilearn_adapter", "gdsfactory_adapter", "netpyne_adapter",
    "Adapter", "AdapterRegistry", "safe", "record_optional",
    "status", "probe_import", "AVAILABLE", "MISSING", "BROKEN",
]
