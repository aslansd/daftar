"""NetPyNE / NEURON adapter.

NetPyNE is the largest install base in computational neuroscience and has the
worst provenance situation in it, for one specific reason.

**Compiled mechanisms are invisible.** NEURON mechanisms are written in NMODL
(``.mod`` files) and compiled by ``nrnivmodl`` into a shared library —
``x86_64/libnrnmech.so`` on Linux, ``arm64/`` on Apple silicon. That binary is
what actually runs. Nothing records which ``.mod`` sources produced it, and a
**stale compiled library silently producing old results** is one of the most
common and most painful failures in the field: you edit a mechanism, forget to
re-run ``nrnivmodl``, and the simulation keeps using the previous binary without
a word. The adapter hashes the ``.mod`` sources *and* the compiled library, and
compares modification times to detect exactly this.

**NetPyNE has its own random seeds.** ``cfg.seeds`` is a dict of four —
``conn``, ``stim``, ``loc``, ``cell`` — feeding NEURON's Random123 streams for
connectivity, stimulation, cell positions and cell parameters. Seeding numpy
does not touch them, and they default to ``1``, so a network that looks
reproducible may be reproducible only by accident. Unlike Brian2's global
``seed()``, these live on a config object rather than a module, so daftar's core
cannot reach them; :func:`seed_config` sets them from the run's seed explicitly.

**The network is generated, not declared.** ``connParams`` with
``probability=0.2`` draws a different graph each time. The realised connection
count is the thing to compare, and it appears nowhere in the parameters.

**`hParams` are global NEURON state.** ``celsius`` defaults to 6.3 °C — a value
inherited from the original squid axon work — and changes every rate constant in
every temperature-dependent mechanism. Most models that should set it do not.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "netpyne"

#: Architecture directories nrnivmodl writes into.
_ARCH_DIRS = ("x86_64", "arm64", "aarch64", "i686", "powerpc", "umac")


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("netpyne")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _sha(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:16]


def _file_sha(path: str) -> str | None:
    def _hash():
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    return safe(_hash)


def _json_sha(obj: Any) -> str | None:
    """Stable hash of a nested parameter structure."""
    def _hash():
        text = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    return safe(_hash)


# --------------------------------------------------------------------------
# compiled mechanisms -- the reason this adapter exists
# --------------------------------------------------------------------------

def describe_mechanisms(run: Run, search_dir: str | None = None,
                        prefix: str = "neuron") -> None:
    """Record NMODL sources, the compiled library, and whether it is stale.

    The staleness check is the point. NEURON loads whatever binary is present;
    if you edited a ``.mod`` file and did not re-run ``nrnivmodl``, the
    simulation runs the *previous* mechanism and says nothing. Comparing
    modification times catches it, and hashing both sides means the manifest can
    prove which sources the binary corresponds to.
    """
    directory = os.path.abspath(search_dir or os.getcwd())
    run.log_param(f"{prefix}.mod_search_dir_sha256", _sha(directory))

    mod_files = sorted(glob.glob(os.path.join(directory, "*.mod")))
    run.log_param(f"{prefix}.n_mod_files", len(mod_files))

    newest_mod = 0.0
    if mod_files:
        digests = {}
        for path in mod_files:
            base = os.path.basename(path)
            digest = _file_sha(path)
            if digest:
                digests[base] = digest
            mtime = safe(lambda p=path: os.path.getmtime(p), 0.0) or 0.0
            newest_mod = max(newest_mod, mtime)
        run.log_param(f"{prefix}.mod_files", sorted(digests))
        # One hash over all sources: the thing the binary should correspond to.
        run.log_param(f"{prefix}.mod_sources_sha256",
                      _sha(sorted(digests.items())))

    # The compiled library that actually runs.
    lib_path = None
    for arch in _ARCH_DIRS:
        candidate = os.path.join(directory, arch, "libnrnmech.so")
        if os.path.isfile(candidate):
            lib_path = candidate
            break
        for pattern in ("libnrnmech.dylib", ".libs/libnrnmech.so"):
            candidate = os.path.join(directory, arch, pattern)
            if os.path.isfile(candidate):
                lib_path = candidate
                break
        if lib_path:
            break

    if lib_path is None:
        run.log_param(f"{prefix}.compiled_mechanisms", False)
        if mod_files:
            run.log_param(
                f"{prefix}.mod_compile_warning",
                "NMODL sources present but no compiled library found; "
                "mechanisms may be loaded from elsewhere",
            )
        return

    run.log_param(f"{prefix}.compiled_mechanisms", True)
    run.log_param(f"{prefix}.compiled_lib_sha256", _file_sha(lib_path) or "unknown")
    run.log_param(f"{prefix}.compiled_lib_bytes",
                  safe(lambda: os.path.getsize(lib_path), "unknown"))
    run.log_param(f"{prefix}.compiled_arch",
                  os.path.basename(os.path.dirname(lib_path)))

    lib_mtime = safe(lambda: os.path.getmtime(lib_path), 0.0) or 0.0
    if mod_files and newest_mod > lib_mtime:
        # The failure this whole function exists to catch.
        run.log_param(f"{prefix}.compiled_lib_stale", True)
        run.log_param(
            f"{prefix}.stale_warning",
            "a .mod source is NEWER than the compiled library: NEURON is "
            "running the previously compiled mechanism. Re-run nrnivmodl.",
        )
    elif mod_files:
        run.log_param(f"{prefix}.compiled_lib_stale", False)


def describe_loaded_mechanisms(run: Run, prefix: str = "neuron") -> None:
    """Record which mechanism names NEURON currently has available."""
    def _names():
        from neuron import h
        mech_type = h.MechanismType(0)
        found = []
        for i in range(int(mech_type.count())):
            ref = h.ref("")
            mech_type.select(i)
            mech_type.selected(ref)
            found.append(ref[0])
        return sorted(found)

    names = safe(_names, [])
    if names:
        run.log_param(f"{prefix}.n_mechanisms_loaded", len(names))
        run.log_param(f"{prefix}.mechanisms_loaded", names)
        run.log_param(f"{prefix}.mechanisms_sha256", _sha(names))


def describe_environment(run: Run, prefix: str = "netpyne") -> None:
    """Record the NetPyNE and NEURON versions, including NEURON's build hash."""
    from importlib.metadata import version

    run.log_param(f"{prefix}.version", safe(lambda: version("netpyne"), "unknown"))
    run.log_param(f"{prefix}.neuron_version",
                  safe(lambda: version("neuron"), "unknown"))

    # NEURON's own string carries the git hash of the build, which the package
    # version does not.
    def _nrn():
        from neuron import h
        return str(h.nrnversion())
    run.log_param(f"{prefix}.neuron_build", safe(_nrn, "unknown"))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def describe_sim_config(cfg: Any, run: Run, prefix: str = "sim") -> None:
    """Record the simulation configuration, seeds and global NEURON state."""
    for attr in ("duration", "dt", "recordStep", "cvode_active", "cvode_atol",
                 "hParams", "verbose", "printPopAvgRates", "recordStim",
                 "allowSelfConns", "connRandomSecFromList"):
        value = safe(lambda a=attr: getattr(cfg, a))
        if value is None:
            continue
        if attr == "hParams":
            # Global NEURON state. celsius defaults to 6.3 degC and changes
            # every temperature-dependent rate constant in the model.
            for key, val in sorted((safe(lambda: dict(value), {}) or {}).items()):
                run.log_param(f"{prefix}.hParams.{key}", val)
        else:
            run.log_param(f"{prefix}.{attr}", value)

    seeds = safe(lambda: dict(cfg.seeds), {}) or {}
    for key, value in sorted(seeds.items()):
        run.log_param(f"{prefix}.seed.{key}", value)
    if seeds:
        run.log_param(f"{prefix}.seeds_all_default",
                      all(v == 1 for v in seeds.values()))

    index = safe(lambda: cfg.rand123GlobalIndex)
    run.log_param(f"{prefix}.rand123GlobalIndex",
                  "unset" if index is None else index)

    record = safe(lambda: dict(cfg.recordTraces), {})
    if record:
        run.log_param(f"{prefix}.recordTraces", sorted(record))


def seed_config(cfg: Any, run: Run, prefix: str = "sim") -> None:
    """Set NetPyNE's four RNG seeds from the daftar run's seed.

    NetPyNE's seeds live on a config object rather than a module, so unlike
    Brian2's global ``seed()`` daftar's core seeding cannot reach them. They
    default to ``1``, which means a network can appear reproducible while the
    reproducibility is accidental -- and stops the moment anyone changes them.

    Called by :func:`run_sim` unless you pass ``seed_from_run=False``.
    """
    base = int(run.seed)
    applied = {}
    for offset, key in enumerate(("conn", "stim", "loc", "cell")):
        value = base + offset
        if safe(lambda k=key, v=value: cfg.seeds.__setitem__(k, v)) is not None \
                or safe(lambda k=key: cfg.seeds[k]) == value:
            applied[key] = value
    if applied:
        run.manifest.set("seed.netpyne", True)
        for key, value in sorted(applied.items()):
            run.manifest.set(f"seed.netpyne_{key}", value)


def describe_net_params(net_params: Any, run: Run, prefix: str = "model") -> None:
    """Record the network specification by structure and content hash.

    The full ``netParams`` is far too large to put in a manifest -- a cortical
    model runs to thousands of lines. Counts plus per-section hashes make a
    change visible and say *which* section changed, which is what a diff needs.
    """
    sections = ("popParams", "cellParams", "synMechParams", "connParams",
                "stimSourceParams", "stimTargetParams", "subConnParams")

    for section in sections:
        value = safe(lambda s=section: getattr(net_params, s))
        if not value:
            continue
        run.log_param(f"{prefix}.n_{section}", safe(lambda: len(value), "unknown"))
        run.log_param(f"{prefix}.{section}_names",
                      safe(lambda: sorted(map(str, value.keys())), []))
        digest = _json_sha(safe(lambda: dict(value)))
        if digest:
            run.log_param(f"{prefix}.{section}_sha256", digest)

    # One hash over the whole specification, so "did the model change at all"
    # is a single field.
    whole = _json_sha(safe(lambda: net_params.todict()))
    if whole:
        run.log_param(f"{prefix}.netParams_sha256", whole)

    for attr in ("scale", "sizeX", "sizeY", "sizeZ", "defaultThreshold",
                 "defaultDelay", "propVelocity"):
        value = safe(lambda a=attr: getattr(net_params, a))
        if value is not None:
            run.log_param(f"{prefix}.{attr}", value)


# --------------------------------------------------------------------------
# the instantiated network and its results
# --------------------------------------------------------------------------

def describe_network(sim_module: Any, run: Run, prefix: str = "network") -> None:
    """Record the network that was actually built.

    ``connParams`` with ``probability=0.2`` draws a different graph on every
    unseeded run. The realised connection count is the number to compare and it
    is nowhere in the parameters.
    """
    cells = safe(lambda: list(sim_module.net.cells), [])
    run.log_result(f"{prefix}.n_cells", len(cells))

    n_conns = safe(lambda: sum(len(c.conns) for c in cells))
    if n_conns is not None:
        run.log_result(f"{prefix}.n_connections", n_conns)
        if cells:
            run.log_result(f"{prefix}.conns_per_cell",
                           round(n_conns / len(cells), 4))

    n_stims = safe(lambda: sum(len(getattr(c, "stims", [])) for c in cells))
    if n_stims is not None:
        run.log_result(f"{prefix}.n_stims", n_stims)

    pops = safe(lambda: dict(sim_module.net.pops), {})
    if pops:
        run.log_param(f"{prefix}.populations", sorted(map(str, pops)))
        for pop_name, pop in sorted(pops.items()):
            count = safe(lambda p=pop: len(p.cellGids), "unknown")
            run.log_result(f"{prefix}.pop.{pop_name}.n_cells", count)

    # Parallel context: results can differ with a different host count.
    run.log_param(f"{prefix}.nhosts", safe(lambda: int(sim_module.nhosts), 1))


def describe_results(sim_module: Any, run: Run, prefix: str = "spikes") -> None:
    """Summarise simulation output as comparable scalars, not traces."""
    data = safe(lambda: sim_module.simData, {}) or {}

    spike_times = safe(lambda: list(data["spkt"]), None)
    if spike_times is not None:
        run.log_result(f"{prefix}.n_spikes", len(spike_times))
        duration = safe(lambda: float(sim_module.cfg.duration))
        n_cells = safe(lambda: len(sim_module.net.cells))
        if duration and n_cells:
            rate = len(spike_times) / (n_cells * duration / 1000.0)
            run.log_result(f"{prefix}.mean_rate_hz", round(rate, 6))

        ids = safe(lambda: list(data["spkid"]), [])
        if ids:
            run.log_result(f"{prefix}.n_active_cells", len(set(ids)))
        if spike_times:
            run.log_result(f"{prefix}.first_spike_ms",
                           round(float(min(spike_times)), 4))
            run.log_result(f"{prefix}.last_spike_ms",
                           round(float(max(spike_times)), 4))

    for key in sorted(data):
        if not key.startswith("V_") and key not in ("t",):
            continue
        def _stats(k=key):
            import numpy as np
            arr = np.asarray([v for v in data[k].values()]
                             if hasattr(data[k], "values") else data[k],
                             dtype="float64")
            return {"mean": round(float(arr.mean()), 8),
                    "min": round(float(arr.min()), 8),
                    "max": round(float(arr.max()), 8)}
        for stat, value in (safe(_stats, {}) or {}).items():
            run.log_result(f"{prefix}.{key}.{stat}", value)


# --------------------------------------------------------------------------
# the wrapper people actually call
# --------------------------------------------------------------------------

def run_sim(net_params: Any, cfg: Any, run: Run, mod_dir: str | None = None,
            seed_from_run: bool = True, analyze: bool = False):
    """Create, simulate and record a NetPyNE model.

    ::

        with daftar.track("cortical-net", seed=42) as run:
            netpyne_adapter.run_sim(netParams, cfg, run)

    Records the compiled-mechanism state *before* running, because that is what
    determines which code executes, and a stale binary is the failure most worth
    catching. Sets NetPyNE's four RNG seeds from the run's seed unless you pass
    ``seed_from_run=False``.
    """
    from netpyne import sim as netpyne_sim

    describe_environment(run)
    describe_mechanisms(run, search_dir=mod_dir)
    describe_loaded_mechanisms(run)

    if seed_from_run:
        seed_config(cfg, run)
    describe_sim_config(cfg, run)
    describe_net_params(net_params, run)

    if analyze:
        netpyne_sim.createSimulateAnalyze(netParams=net_params, simConfig=cfg)
    else:
        netpyne_sim.createSimulate(netParams=net_params, simConfig=cfg)

    describe_network(netpyne_sim, run)
    describe_results(netpyne_sim, run)
    return netpyne_sim


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever NetPyNE object is handed in."""
    cls = type(obj).__name__
    if cls == "SimConfig":
        describe_sim_config(obj, run, prefix or "sim")
    elif cls == "NetParams":
        describe_net_params(obj, run, prefix or "model")
    elif hasattr(obj, "net") and hasattr(obj, "simData"):
        describe_network(obj, run, prefix or "network")
        describe_results(obj, run)
    else:
        run.log_param(f"{prefix or 'netpyne'}.type", cls)
