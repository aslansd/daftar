"""Brian2 adapter.

Brian2 is the strongest case in this package for a domain adapter, because more
of what determines the answer is invisible than in any other framework here.

**The integration method is usually chosen for you and then forgotten.** The
default is not a method but a list of candidates, ``('exact', 'euler', 'heun')``,
tried in order until one fits the equations. Which one won depends on the
equations, so a model that integrated exactly can silently fall back to Euler
after an edit that makes the system non-linear. Brian2 reports the choice *only*
as a log line -- ``group_name.state_updater.method_choice`` still holds the list
you never chose from. This adapter captures the log during ``run()`` and records
the method that actually ran, per group. Nothing else records it.

**The code-generation target changes numerics.** ``prefs.codegen.target``
defaults to ``auto``, which resolves to Cython where a compiler exists and NumPy
where one does not. The same script on a laptop and on a cluster can therefore
take different code paths, and neither the script nor the resolved value appears
anywhere afterwards.

**Brian2 has its own RNG.** ``brian2.seed()`` delegates to the current device;
seeding ``numpy`` does not seed it. daftar's core seeding now calls it, and the
manifest records that it did.

**The schedule is an experiment parameter.** ``net.schedule`` orders
thresholds, synapses and resets within a timestep. Reordering it changes results
without changing a line of model code.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from ..run import Run
from .base import safe

name = "brian2"

#: ``No numerical integration method specified for group 'G', using method
#: 'exact' (took 0.02s).`` -- and the shorter ``Group G: using numerical
#: integration method exact``. Both are matched; the first is authoritative.
_METHOD_PATTERNS = (
    re.compile(r"for group '(?P<group>[^']+)', using method '(?P<method>[^']+)'"),
    re.compile(r"Group (?P<group>\S+): using numerical integration method "
               r"(?P<method>\S+)"),
)


def is_available() -> bool:
    try:
        import brian2  # noqa: F401
        return True
    except Exception:
        return False


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------
# capturing the resolved integration method
# --------------------------------------------------------------------------

class _MethodCapture(logging.Handler):
    """Collects ``group -> resolved method`` from Brian2's own log output.

    Reading a log is not elegant, but it is the only route: Brian2 resolves the
    method during ``before_run`` and does not store the winner on any object.
    The handler is attached for the duration of the run and removed afterwards,
    and it never raises -- a logging handler that throws would take the
    simulation down with it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.methods: dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        for pattern in _METHOD_PATTERNS:
            m = pattern.search(message)
            if m:
                self.methods.setdefault(m.group("group"), m.group("method"))
                return


class capture_methods:
    """Attach :class:`_MethodCapture` to Brian2's logger for a block.

    No longer used by :func:`run_network` -- :func:`resolve_method` reproduces
    the selection directly, which is immune to Brian2's caching of
    ``apply_stateupdater``. Kept as a public escape hatch for anyone who wants
    the method as Brian2 itself reported it, and for cross-checking that our
    resolution agrees with Brian's.
    """

    def __init__(self) -> None:
        self.handler = _MethodCapture()
        self._logger = logging.getLogger("brian2")
        self._previous_level: int | None = None

    def __enter__(self) -> _MethodCapture:
        self._previous_level = self._logger.level
        # Brian2 emits the method choice at DEBUG. Lower the level only if it
        # would otherwise suppress the message, and restore it on exit.
        if self._logger.level > logging.DEBUG or self._logger.level == logging.NOTSET:
            self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self.handler)
        return self.handler

    def __exit__(self, *exc) -> bool:
        self._logger.removeHandler(self.handler)
        if self._previous_level is not None:
            self._logger.setLevel(self._previous_level)
        return False


# --------------------------------------------------------------------------
# preferences, device, clock
# --------------------------------------------------------------------------

#: Preferences that change results and are almost never written down.
_TRACKED_PREFS = (
    "codegen.target",
    "codegen.loop_invariant_optimisations",
    "codegen.string_expression_target",
    "core.default_float_dtype",
    "core.network.default_schedule",
    "legacy.refractory_timing",
)


def describe_environment(run: Run, prefix: str = "brian2") -> None:
    """Record the Brian2 preferences and device that shape every result."""
    import brian2

    run.log_param(f"{prefix}.version", safe(lambda: brian2.__version__, "unknown"))

    prefs = safe(lambda: brian2.prefs)
    if prefs is not None:
        for key in _TRACKED_PREFS:
            value = safe(lambda k=key: prefs[k])
            if value is None:
                continue
            # core.default_float_dtype is a numpy type object; its repr is a
            # class path, so reduce it to the name that actually matters.
            run.log_param(f"{prefix}.prefs.{key}",
                          getattr(value, "__name__", None) or value)

    device = safe(lambda: brian2.get_device())
    if device is not None:
        run.log_param(f"{prefix}.device", type(device).__name__)

    # `codegen.target = auto` resolves to cython where a compiler exists and
    # numpy where it does not, so the *resolved* value is the one that matters.
    run.log_param(f"{prefix}.codegen_resolved", safe(_resolved_target, "unknown"))

    clock = safe(lambda: brian2.defaultclock)
    if clock is not None:
        run.log_param(f"{prefix}.defaultclock_dt", _quantity(clock.dt))


def resolve_method(group: Any) -> str | None:
    """Which integration method Brian2 will actually use for ``group``.

    Brian2's default is not a method but a candidate list,
    ``('exact', 'euler', 'heun')``, tried in order until one accepts the
    equations. The winner is stored nowhere: ``state_updater.method_choice``
    still holds the list.

    Brian2 does log the choice, but ``apply_stateupdater`` is cached, so the
    line is emitted only the first time a given set of equations is seen in a
    process. Reading the log therefore records the method on run 1 and nothing
    on run 2 -- which makes the field *appear* and *disappear* between runs and
    shows up as a spurious cause in every diff. Reproducing the selection
    directly is deterministic and immune to the cache.
    """
    from brian2.stateupdaters.base import (
        StateUpdateMethod, UnsupportedEquationsException,
    )

    updater = safe(lambda: group.state_updater)
    if updater is None:
        return None
    choice = safe(lambda: updater.method_choice)
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if callable(choice):
        return getattr(choice, "__name__", None) or type(choice).__name__

    equations = safe(lambda: group.equations)
    variables = safe(lambda: group.variables)
    if equations is None or variables is None:
        return None

    for candidate in choice:
        updater_impl = safe(lambda c=candidate: StateUpdateMethod.stateupdaters[c])
        if updater_impl is None:
            continue
        try:
            updater_impl(equations, variables)
            return str(candidate)
        except UnsupportedEquationsException:
            continue
        except Exception:
            # An unrelated failure tells us nothing about suitability; keep
            # looking rather than reporting a method that may not be the one.
            continue
    return None


def _resolved_target() -> str:
    """What ``codegen.target = 'auto'`` actually became.

    ``auto`` resolves to Cython where a working compiler exists and NumPy where
    one does not, so the same script takes different code paths on a laptop and
    on a cluster. Neither the script nor the preference records which happened.
    """
    import brian2

    target = safe(lambda: brian2.prefs["codegen.target"], "unknown")
    if target != "auto":
        return str(target)
    try:
        from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
        if CythonCodeObject.is_available():
            return "cython (auto)"
    except Exception:
        pass
    return "numpy (auto)"


def _quantity(value: Any) -> str:
    """Render a Brian2 Quantity in a stable, unit-bearing form.

    ``str(100*usecond)`` gives ``'100. us'``. Keeping the unit is the point:
    a manifest recording ``dt = 0.0001`` loses the fact that it is seconds, and
    a bare float invites the reader to guess.
    """
    return safe(lambda: str(value), "<unavailable>")


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------

def describe_group(group: Any, run: Run, prefix: str | None = None) -> None:
    """Record a NeuronGroup: size, equations, threshold, method choice, clock."""
    gname = safe(lambda: group.name, "group")
    prefix = prefix or f"group.{gname}"

    run.log_param(f"{prefix}.type", type(group).__name__)
    run.log_param(f"{prefix}.N", safe(lambda: int(group.N), "unknown"))

    # The equations *are* the model. Hash them so an edit is visible, and record
    # the variable names so a diff says which part changed.
    eqs = safe(lambda: str(group.equations))
    if eqs:
        run.log_param(f"{prefix}.equations_sha256", _sha(eqs))
        run.log_param(f"{prefix}.diff_eq_names",
                      safe(lambda: sorted(group.equations.diff_eq_names), []))
        run.log_param(f"{prefix}.parameter_names",
                      safe(lambda: sorted(group.equations.parameter_names), []))

    for attr, key in (("_threshold", "threshold"), ("_reset", "reset"),
                      ("_refractory", "refractory")):
        value = safe(lambda a=attr: getattr(group, a))
        if value not in (None, False):
            run.log_param(f"{prefix}.{key}", _quantity(value))

    updater = safe(lambda: group.state_updater)
    if updater is not None:
        # method_choice is the candidate *list*; method_resolved is the winner.
        # Both are recorded: the list shows what was permitted, the winner shows
        # what ran. A model that integrated exactly and silently fell back to
        # euler after an edit shows up as a change in the second field only.
        run.log_param(f"{prefix}.method_choice",
                      _quantity(safe(lambda: updater.method_choice)))
        run.log_param(f"{prefix}.method_resolved",
                      safe(lambda: resolve_method(group), "unknown") or "unknown")
        options = safe(lambda: updater.method_options)
        if options:
            run.log_param(f"{prefix}.method_options", options)

    clock = safe(lambda: group.clock)
    if clock is not None:
        run.log_param(f"{prefix}.dt", _quantity(clock.dt))

    events = safe(lambda: dict(group.events), {})
    if events:
        run.log_param(f"{prefix}.events", sorted(events.keys()))


def describe_synapses(syn: Any, run: Run, prefix: str | None = None) -> None:
    """Record a Synapses object: connectivity, delays, equations.

    The number of realised synapses matters and is not knowable from the script:
    ``connect(p=0.02)`` draws a different graph on every unseeded run, which is
    one of the commonest sources of "why is my network different today".
    """
    sname = safe(lambda: syn.name, "synapses")
    prefix = prefix or f"synapses.{sname}"

    run.log_param(f"{prefix}.type", type(syn).__name__)
    run.log_param(f"{prefix}.n_synapses", safe(lambda: int(len(syn)), "unknown"))
    run.log_param(f"{prefix}.source", safe(lambda: syn.source.name, "unknown"))
    run.log_param(f"{prefix}.target", safe(lambda: syn.target.name, "unknown"))

    eqs = safe(lambda: str(syn.equations))
    if eqs:
        run.log_param(f"{prefix}.equations_sha256", _sha(eqs))

    for attr, key in (("_on_pre", "on_pre"), ("_on_post", "on_post")):
        value = safe(lambda a=attr: getattr(syn, a))
        if value:
            run.log_param(f"{prefix}.{key}_sha256", _sha(str(value)))

    delay = safe(lambda: syn.delay)
    if delay is not None:
        run.log_param(f"{prefix}.delay_summary", _delay_summary(delay))


def _delay_summary(delay: Any) -> str:
    """min/max of a delay array, or the scalar. Recording every delay is noise."""
    def _summary():
        import numpy as np
        arr = np.asarray(delay[:])
        if arr.size == 0:
            return "none"
        if arr.size == 1 or arr.min() == arr.max():
            return f"{float(arr.flat[0]):.6g} s (uniform)"
        return f"{float(arr.min()):.6g}..{float(arr.max()):.6g} s"
    return safe(_summary, "<unavailable>")


def describe_network(net: Any, run: Run, prefix: str = "network") -> None:
    """Record the network: which objects, and the within-timestep schedule."""
    objects = safe(lambda: sorted(o.name for o in net.objects), [])
    run.log_param(f"{prefix}.objects", objects)
    run.log_param(f"{prefix}.n_objects", len(objects))

    # Reordering the schedule changes results without changing model code.
    run.log_param(f"{prefix}.schedule", safe(lambda: list(net.schedule), []))
    run.log_param(f"{prefix}.t_start", _quantity(safe(lambda: net.t)))

    for obj in safe(lambda: list(net.objects), []):
        cls = type(obj).__name__
        if cls in ("NeuronGroup", "PoissonGroup", "SpikeGeneratorGroup"):
            describe_group(obj, run)
        elif cls == "Synapses":
            describe_synapses(obj, run)


# --------------------------------------------------------------------------
# monitors -> results
# --------------------------------------------------------------------------

def describe_monitors(net: Any, run: Run, prefix: str = "monitor") -> None:
    """Summarise monitor output as comparable scalars.

    Deliberately summaries, not traces. A recorded voltage array would make
    every diff unreadable; spike counts and rates are what you actually compare
    between two runs.
    """
    for obj in safe(lambda: list(net.objects), []):
        cls = type(obj).__name__
        oname = safe(lambda o=obj: o.name, "monitor")

        if cls == "SpikeMonitor":
            run.log_result(f"{prefix}.{oname}.num_spikes",
                           safe(lambda o=obj: int(o.num_spikes), "unknown"))
            run.log_result(f"{prefix}.{oname}.n_active_neurons",
                           safe(lambda o=obj: int(len(set(o.i[:]))), "unknown"))

        elif cls == "PopulationRateMonitor":
            def _mean_rate(o=obj):
                import numpy as np
                return round(float(np.asarray(o.rate[:]).mean()), 6)
            run.log_result(f"{prefix}.{oname}.mean_rate_hz",
                           safe(_mean_rate, "unknown"))

        elif cls == "StateMonitor":
            variables = safe(lambda o=obj: list(o.record_variables), [])
            run.log_param(f"{prefix}.{oname}.variables", sorted(variables))
            for var in variables:
                def _stats(o=obj, v=var):
                    import numpy as np
                    arr = np.asarray(getattr(o, v)[:])
                    return {
                        "mean": round(float(arr.mean()), 8),
                        "min": round(float(arr.min()), 8),
                        "max": round(float(arr.max()), 8),
                        "n_nonfinite": int((~np.isfinite(arr)).sum()),
                    }
                for stat, value in (safe(_stats, {}) or {}).items():
                    run.log_result(f"{prefix}.{oname}.{var}.{stat}", value)


# --------------------------------------------------------------------------
# the wrapper people actually call
# --------------------------------------------------------------------------

def run_network(net: Any, duration: Any, run: Run, **kwargs: Any):
    """``net.run(duration)`` with everything provenance-relevant recorded.

    ::

        with daftar.track("balanced-network", seed=42) as run:
            brian2_adapter.run_network(net, 1*second, run)

    The resolved integration method is captured from Brian2's log during the
    run, because it exists nowhere else afterwards.
    """
    describe_environment(run)
    describe_network(net, run)
    run.log_param("brian2.run_duration", _quantity(duration))

    result = net.run(duration, **kwargs)

    describe_monitors(net, run)
    run.log_result("brian2.t_end", _quantity(safe(lambda: net.t)))
    return result


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever Brian2 object is handed in."""
    cls = type(obj).__name__
    if cls == "Network":
        describe_network(obj, run)
    elif cls == "Synapses":
        describe_synapses(obj, run, prefix)
    else:
        describe_group(obj, run, prefix)
