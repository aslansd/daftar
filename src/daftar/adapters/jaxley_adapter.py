"""Jaxley adapter.

What a generic tracker misses about a Jaxley run, and what this records instead:

* ``jx.integrate`` takes ``solver`` and ``voltage_solver`` with *defaults*
  (``bwd_euler``, ``jaxley.dhs``). Someone who never passes them has no record
  of them, and Jaxley changing a default between versions silently moves every
  result. Defaults are recorded explicitly.
* ``delta_t`` defaults to 0.025 ms and is the single most common cause of a
  result moving. Recorded whether or not it was passed.
* The morphology is provenance. A ``Cell`` built from an SWC file, its branch
  and compartment counts, and which channels are inserted where, all determine
  the answer and none of it appears in the call arguments.
* Jaxley is JAX, so x64 mode matters. A run made with
  ``jax_enable_x64=True`` and one without are different experiments that look
  identical in any generic log.
"""

from __future__ import annotations

import sys
from typing import Any

from ..run import Run
from .base import safe

name = "jaxley"

#: jx.integrate defaults, as of Jaxley 0.6.x. Recorded when not passed
#: explicitly, so that a future change of default is visible as a diff.
INTEGRATE_DEFAULTS = {
    "delta_t": 0.025,
    "solver": "bwd_euler",
    "voltage_solver": "jaxley.dhs",
    "t_max": None,
    "checkpoint_lengths": None,
    "return_states": False,
}


def is_available() -> bool:
    try:
        import jaxley  # noqa: F401
        return True
    except Exception:
        return False


def _jax_config(run: Run) -> None:
    jax = sys.modules.get("jax")
    if jax is None:
        return
    cfg = getattr(jax, "config", None)
    if cfg is None:
        return
    # x64 silently changes numerical results; platform changes them too.
    run.manifest.set("env.jax_enable_x64", safe(lambda: cfg.jax_enable_x64, "unknown"))
    run.manifest.set(
        "env.jax_platform",
        safe(lambda: jax.default_backend(), "unknown"),
    )
    run.manifest.set(
        "env.jax_devices",
        safe(lambda: ",".join(sorted({d.platform for d in jax.devices()})), "unknown"),
    )


def describe_module(module: Any, run: Run, prefix: str = "morphology") -> None:
    """Record the structure of a ``jx.Module`` (Cell, Network, Branch, Compartment)."""
    run.log_param(f"{prefix}.type", type(module).__name__)

    # Compartment/branch/cell counts live in the nodes DataFrame.
    nodes = safe(lambda: module.nodes)
    if nodes is not None:
        run.log_param(f"{prefix}.n_compartments", safe(lambda: len(nodes), "unknown"))
        for col, label in (
            ("global_branch_index", "n_branches"),
            ("global_cell_index", "n_cells"),
        ):
            run.log_param(
                f"{prefix}.{label}",
                safe(lambda c=col: int(nodes[c].nunique()), "unknown"),
            )

        # Which channels are inserted, and how many compartments carry each.
        channels = safe(lambda: [c._name for c in module.channels], [])
        if channels:
            run.log_param(f"{prefix}.channels", sorted(channels))
            for ch in sorted(set(channels)):
                run.log_param(
                    f"{prefix}.channel_compartments.{ch}",
                    safe(lambda c=ch: int(nodes[c].sum()) if c in nodes else 0, "unknown"),
                )

    edges = safe(lambda: module.edges)
    if edges is not None and len(edges):
        run.log_param(f"{prefix}.n_synapses", len(edges))
        run.log_param(
            f"{prefix}.synapse_types",
            safe(lambda: sorted(set(edges["type"].tolist())), "unknown"),
        )

    # Trainable parameters change what a gradient step does.
    run.log_param(
        f"{prefix}.n_trainable_params",
        safe(lambda: int(module.num_trainable_params), 0),
    )
    # Externals: stimuli and clamps are inputs, not incidental.
    externals = safe(lambda: module.externals, {})
    if externals:
        run.log_param(f"{prefix}.externals", sorted(externals.keys()))
        for k, v in sorted(externals.items()):
            run.log_param(
                f"{prefix}.external.{k}.shape",
                safe(lambda vv=v: str(tuple(vv.shape)), "unknown"),
            )


def describe_integration(run: Run, **kwargs: Any) -> dict[str, Any]:
    """Record integrator settings, filling in Jaxley's defaults explicitly.

    Returns the fully-resolved kwargs so the caller can pass them straight to
    ``jx.integrate`` -- there is then no way for what was recorded and what was
    run to drift apart.
    """
    resolved = dict(INTEGRATE_DEFAULTS)
    resolved.update({k: v for k, v in kwargs.items() if v is not None or k in resolved})

    for key, value in resolved.items():
        run.log_param(f"integrate.{key}", value)
        if key in INTEGRATE_DEFAULTS and key not in kwargs:
            run.log_param(f"integrate.{key}.was_default", True)

    _jax_config(run)
    return resolved


def integrate(module: Any, run: Run, **kwargs: Any):
    """``jx.integrate`` with the module and settings recorded first.

    ::

        with daftar.track("celegans", seed=0) as run:
            v = jaxley_adapter.integrate(cell, run, t_max=10.0, delta_t=0.025)
    """
    import jaxley as jx

    describe_module(module, run)
    resolved = describe_integration(run, **kwargs)
    run.log_param("integrate.jaxley_version", safe(lambda: jx.__version__, "unknown"))

    call_kwargs = {k: v for k, v in resolved.items() if v is not None}
    result = jx.integrate(module, **call_kwargs)

    # Summary statistics, not the trace. A recorded array would make every diff
    # unreadable; these four numbers are what you actually compare.
    def _stats():
        import numpy as np
        arr = np.asarray(result)
        return {
            "shape": str(arr.shape),
            "v_mean": float(arr.mean()),
            "v_min": float(arr.min()),
            "v_max": float(arr.max()),
            "v_final_mean": float(arr[..., -1].mean()),
            "n_nonfinite": int((~np.isfinite(arr)).sum()),
        }

    stats = safe(_stats, {})
    for k, v in stats.items():
        run.log_result(f"voltage.{k}", v)
    return result


def describe(obj: Any, run: Run, prefix: str = "morphology") -> None:
    describe_module(obj, run, prefix)
