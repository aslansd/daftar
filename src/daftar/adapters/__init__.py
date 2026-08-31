"""Framework adapters.

Import is lazy on purpose: someone with Jaxley but not MeltingPot installed must
still be able to ``import daftar``. Nothing here imports a target framework
at module load.
"""

from __future__ import annotations

from . import cpm_adapter, jaxley_adapter, meltingpot_adapter
from .base import Adapter, AdapterRegistry, record_optional, safe

registry = AdapterRegistry()
registry.register("jaxley", jaxley_adapter)
registry.register("cpm", cpm_adapter)
registry.register("meltingpot", meltingpot_adapter)

jaxley = jaxley_adapter
cpm = cpm_adapter
meltingpot = meltingpot_adapter


def available() -> list[str]:
    """Adapters whose target framework is importable right now."""
    return registry.available()


def get(name: str):
    return registry.get(name)


__all__ = [
    "registry", "available", "get",
    "jaxley", "cpm", "meltingpot",
    "jaxley_adapter", "cpm_adapter", "meltingpot_adapter",
    "Adapter", "AdapterRegistry", "safe", "record_optional",
]
