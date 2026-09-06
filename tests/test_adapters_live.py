"""Live adapter tests. Each skips cleanly when its framework is absent.

These are separated from ``test_core.py`` because they need heavy optional
dependencies and are slow. They are also the only tests that can catch adapter
rot: adapters read attributes of fast-moving research code, and the failure mode
is silent -- a renamed attribute produces ``<unavailable>`` in the manifest
rather than an exception, by design. Only a live run against a real object
notices.

    pip install -e ".[dev]" jaxley cpm-toolbox
    pytest tests/test_adapters_live.py -v

Run this after every upgrade of a target framework, not just at release.
"""

from __future__ import annotations

import pytest

import daftar
from daftar.store import RunStore

jaxley = pytest.importorskip
_ = jaxley  # keep the name from confusing linters


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAFTAR_DIR", str(tmp_path / ".daftar"))
    s = RunStore()
    s.init()
    return s


def _require(name: str):
    """Skip with a reason that distinguishes absent from broken.

    A framework that is installed but fails to import is a different situation
    from one that was never installed, and the user can act on it. Reporting
    both as "not installed" is how a broken environment goes unnoticed: the
    suite skips quietly and nothing anywhere says why.
    """
    state, reason = daftar.adapters.get(name).availability()
    if state == daftar.adapters.AVAILABLE:
        return pytest.mark.skipif(False, reason="")
    if state == daftar.adapters.BROKEN:
        return pytest.mark.skipif(
            True,
            reason=(f"{name} is INSTALLED BUT BROKEN -- {reason}. "
                    f"Run `daftar doctor`; see TROUBLESHOOTING.md"),
        )
    return pytest.mark.skipif(True, reason=f"{name} not installed")


def _require_working(name: str, probe, clash_marker: str = "version clash"):
    """Skip only for a *known* framework/dependency clash; fail on anything else.

    This is the third instance of the same pattern -- jaxley against JAX,
    brian2 against NumPy, cpm against SciPy -- so it is worth stating the rule
    once. Research packages pin loosely and their dependencies remove APIs, so
    a framework that imports cleanly can still be unusable.

    The rule: skip when we recognise the clash and can explain it, because it is
    not our bug and the user needs to be told what to do. Fail loudly for
    anything else, because an unrecognised error may well be *our* bug and a
    silent skip would hide it.
    """
    state, reason = daftar.adapters.get(name).availability()
    if state != daftar.adapters.AVAILABLE:
        return pytest.mark.skipif(True, reason=f"{name}: {reason}")
    ok, why = probe()
    return pytest.mark.skipif(not ok and clash_marker in why, reason=why)


def _cpm_is_usable() -> tuple[bool, str]:
    """Check that cpm can actually fit before blaming the adapter.

    cpm 0.25.6 calls ``fmin_l_bfgs_b(..., disp=self.display)``. SciPy deprecated
    ``disp`` and ``iprint`` for L-BFGS-B and removed them in 1.18.0, so cpm
    imports fine and then raises ``TypeError`` the moment you fit anything.
    ``daftar doctor`` reports cpm as ``ok`` because the import succeeds --
    importing is not the same as working.
    """
    try:
        import numpy as np
        import pandas as pd
        from cpm.generators import Parameters, Value, Wrapper
        from cpm.optimisation import FminBound, minimise

        def model(parameters, trial):
            return {"dependent": np.array([trial["stimulus"] * parameters.alpha])}

        data = pd.DataFrame({
            "ppt": [1, 1, 2, 2],
            "stimulus": [1.0, 2.0, 1.0, 2.0],
            "observed": [0.1, 0.2, 0.15, 0.25],
        })
        params = Parameters(alpha=Value(value=0.1, lower=0.0, upper=1.0,
                                        prior="norm",
                                        args={"mean": 0.5, "sd": 0.25}))
        wrapper = Wrapper(model=model, data=data.iloc[:2], parameters=params)
        FminBound(model=wrapper, data=data,
                  minimisation=minimise.LogLikelihood.continuous,
                  ppt_identifier="ppt", approx_grad=True).optimise()
        return True, ""
    except TypeError as exc:
        if "disp" in str(exc) or "iprint" in str(exc):
            import scipy
            return False, (
                f"cpm/SciPy version clash: this cpm release passes disp= to "
                f"fmin_l_bfgs_b, which SciPy {scipy.__version__} removed in "
                f"1.18.0. Not a daftar bug. Fix with: pip install 'scipy<1.18'"
            )
        return False, f"cpm unusable: {type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"cpm unusable: {type(exc).__name__}: {exc}"


def _jaxley_is_usable() -> tuple[bool, str]:
    """Check that Jaxley can integrate at all before blaming the adapter.

    Released Jaxley 0.13.0 calls ``jnp.clip(x, a_max=...)`` in
    ``solver_gate.save_exp``. JAX deprecated ``a_min``/``a_max`` and later
    removed them, so Jaxley 0.13.0 + a current JAX raises ``TypeError`` deep
    inside the HH channel. That is a version incompatibility between two other
    packages; the adapter is fine. Without this preflight the failure surfaces
    as a confusing traceback in our test suite and looks like our bug.

    Jaxley's ``main`` has fixed it (``jnp.minimum``), so the remedy is to
    install from git or pin JAX.
    """
    try:
        import jaxley as jx
        from jaxley.channels import HH

        cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
        cell.insert(HH())
        cell.branch(0).loc(0.0).record(verbose=False)
        jx.integrate(cell, t_max=0.1)
        return True, ""
    except TypeError as exc:
        if "a_max" in str(exc) or "a_min" in str(exc):
            return False, (
                "Jaxley/JAX version clash: this Jaxley release calls "
                "jnp.clip(a_max=...), which current JAX removed. Not a daftar "
                "bug. Fix with: pip install "
                "'jaxley @ git+https://github.com/jaxleyverse/jaxley.git' "
                "or pin an older JAX."
            )
        return False, f"Jaxley unusable: {type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"Jaxley unusable: {type(exc).__name__}: {exc}"


def _require_working_jaxley():
    """Skip only for the known, benign Jaxley/JAX version clash.

    Any *other* preflight failure lets the test run and fail loudly. A silent
    skip on an unrecognised error is worse than a red test: the adapter could
    be genuinely broken and nothing would say so. Skips are for "this is not
    our problem and we know why", not for "something went wrong".
    """
    if not daftar.adapters.get("jaxley").is_available():
        return pytest.mark.skipif(True, reason="jaxley not installed")
    ok, why = _jaxley_is_usable()
    benign = not ok and "version clash" in why
    return pytest.mark.skipif(benign, reason=why)


# ==========================================================================
# Jaxley
# ==========================================================================

@_require_working_jaxley()
def test_jaxley_records_morphology_and_solver_defaults(store):
    import jaxley as jx
    from jaxley.channels import HH

    from daftar.adapters import jaxley as jxa

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.branch(0).loc(0.0).record()
    cell.branch(0).loc(0.0).stimulate(
        jx.step_current(1.0, 2.0, 0.1, 0.025, 10.0)
    )

    with daftar.track("hh-cell", seed=0) as run:
        v = jxa.integrate(cell, run, t_max=10.0)
        rid = run.run_id

    m = store.load(rid)

    # Morphology: none of this appears in the integrate() call.
    assert m.get("param.morphology.type") == "Cell"
    assert int(m.get("param.morphology.n_compartments")) == 12
    assert int(m.get("param.morphology.n_branches")) == 3
    assert "HH" in m.get("param.morphology.channels")

    # Defaults the caller never passed must still be recorded, and flagged as
    # defaults -- so a future Jaxley release changing one shows up as a diff.
    assert m.get("param.integrate.delta_t") == "0.025"
    assert m.get("param.integrate.solver") == "bwd_euler"
    assert m.get("param.integrate.solver.was_default") == "true"
    assert m.get("param.integrate.voltage_solver.was_default") == "true"
    # t_max was passed, so it is not a default.
    assert m.get("param.integrate.t_max.was_default") is None

    # JAX configuration silently changes numerics.
    assert m.get("env.jax_enable_x64") is not None
    assert m.get("env.jax_platform") is not None

    # Result summary, not the trace.
    assert m.get("result.voltage.v_mean") is not None
    assert m.get("result.voltage.n_nonfinite") == "0"
    assert v is not None


@_require_working_jaxley()
def test_jaxley_same_config_reproduces(store):
    """Two identical Jaxley runs must diff to `identical`."""
    import jaxley as jx
    from jaxley.channels import HH

    from daftar.adapters import jaxley as jxa

    def build():
        cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=2), parents=[-1])
        cell.insert(HH())
        cell.branch(0).loc(0.0).record()
        return cell

    ids = []
    for _ in range(2):
        with daftar.track("repro", seed=0) as run:
            jxa.integrate(build(), run, t_max=5.0)
            ids.append(run.run_id)

    d = daftar.diff_manifests(store.load(ids[0]), store.load(ids[1]))
    assert not d.effects, f"Jaxley run did not reproduce: {d.effects}"


# ==========================================================================
# cpm
# ==========================================================================

@_require_working("cpm", _cpm_is_usable)
def test_cpm_records_bounds_priors_and_restarts(store):
    import numpy as np
    import pandas as pd
    from cpm.generators import Parameters, Value, Wrapper
    from cpm.optimisation import FminBound, minimise

    from daftar.adapters import cpm as cpma

    def model(parameters, trial):
        return {"dependent": np.array([trial["stimulus"] * parameters.alpha])}

    data = pd.DataFrame({
        "ppt": [1, 1, 1, 2, 2, 2],
        "stimulus": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "observed": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
    })
    # cpm's Parameters.free() returns only parameters that have a prior, and
    # FminBound refuses to run a model with no free parameters. A Value with
    # bounds but no prior is fixed, not free -- which is itself the sort of
    # unstated assumption the cpm authors wrote their paper about.
    parameters = Parameters(
        alpha=Value(
            value=0.1, lower=0.0, upper=1.0,
            prior="norm", args={"mean": 0.5, "sd": 0.25},
        )
    )
    wrapper = Wrapper(model=model, data=data.iloc[:3], parameters=parameters)

    # approx_grad=True is required and lands in **kwargs. cpm forwards kwargs
    # straight to scipy's fmin_l_bfgs_b, which without it defaults to
    # approx_grad=0 and expects the objective to return (value, gradient) --
    # cpm's returns a scalar, so scipy raises deep inside MemoizeJac.
    #
    # This is precisely the class of setting the adapter exists to record: a
    # numerical choice that lives in an opaque kwargs dict, changes the answer,
    # and appears nowhere in the model definition.
    fit = FminBound(
        model=wrapper, data=data, minimisation=minimise.LogLikelihood.continuous,
        ppt_identifier="ppt", number_of_starts=2, approx_grad=True,
    )

    with daftar.track("bandit-fit", seed=7) as run:
        cpma.optimise(fit, run)
        rid = run.run_id

    m = store.load(rid)

    assert m.get("param.fit.estimator") == "FminBound"
    # Bounds are the model. Two fits with different bounds are different
    # experiments even with identical code and data.
    assert m.get("param.model.bounds.alpha") == "[0.0, 1.0]"
    assert m.get("param.model.n_parameters") == "1"
    assert m.get("param.model.n_free_parameters") == "1"
    assert m.get("param.model.prior.alpha") is not None

    # cpm discards number_of_starts; it survives only as the first dimension of
    # initial_guess. The adapter recovers it the same way cpm does internally.
    assert m.get("param.fit.number_of_starts") == "2"
    # The guesses themselves are recorded. When drawn at random they differ
    # between runs, so a diff attributes a changed fit to them rather than
    # calling it nondeterministic.
    guesses = m.get("param.fit.initial_guess")
    assert guesses is not None and guesses.startswith("[["), guesses

    # Cohort, without putting subject identifiers in a committed file.
    assert m.get("param.data.n_participants") == "2"
    assert len(m.get("param.data.participant_id_sha256")) == 12

    # Convergence, not just the estimate.
    assert m.get("result.fit.n_fits") is not None
    assert m.get("result.fit.n_converged") is not None

    # Opaque scipy kwargs must survive into the manifest.
    assert m.get("param.fit.kwargs.approx_grad") == "true"


@_require("cpm")
def test_cpm_parameters_alone_can_be_described(store):
    from cpm.generators import Parameters, Value

    from daftar.adapters import cpm as cpma

    parameters = Parameters(
        alpha=Value(value=0.5, lower=0.0, upper=1.0,
                    prior="norm", args={"mean": 0.5, "sd": 0.1}),
        beta=Value(value=1.0, lower=0.0, upper=10.0),
    )
    with daftar.track("params") as run:
        cpma.describe(parameters, run)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.model.n_parameters") == "2"
    assert "alpha" in m.get("param.model.parameter_names")
    # beta has no prior, so it is fixed rather than free. Recording both counts
    # separately is the point: "2 parameters" and "1 free parameter" are
    # different facts and only one of them is in the model's own repr.
    assert m.get("param.model.n_free_parameters") == "1"


# ==========================================================================
# MeltingPot
# ==========================================================================

@_require("meltingpot")
def test_meltingpot_records_config_roles_and_returns(store):
    import numpy as np

    from daftar.adapters import meltingpot as mpa

    roles = ["default"] * 2

    def random_policy(timestep, player):
        return 0  # NOOP; deterministic so the test is stable

    with daftar.track("commons", seed=1234) as run:
        substrate = mpa.build("commons_harvest__open", roles, run)
        stats = mpa.run_episode(substrate, random_policy, run, max_steps=20)
        rid = run.run_id
        substrate.close()

    m = store.load(rid)

    # A mutated ConfigDict is invisible from the substrate name alone.
    assert len(m.get("param.substrate.config_sha256")) == 16
    assert m.get("param.substrate.n_players") == "2"
    assert m.get("param.substrate.name") == "commons_harvest__open"

    # Per-player returns, not just the total: the distribution is the finding.
    assert m.get("result.episode.return.player_0") is not None
    assert m.get("result.episode.return.player_1") is not None
    assert m.get("result.episode.gini") is not None
    assert int(m.get("result.episode.steps")) <= 20
    assert len(stats["returns"]) == 2
    assert np is not None


@_require("meltingpot")
def test_meltingpot_scenario_records_bot_checkpoints(store):
    from daftar.adapters import meltingpot as mpa

    with daftar.track("scenario") as run:
        mpa.describe_scenario("commons_harvest__open_0", run)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.scenario.substrate") is not None
    # Which bot checkpoints faced your agent is part of the result.
    assert m.get("param.scenario.bots") is not None
    assert m.get("param.scenario.n_focal") is not None


# ==========================================================================
# cross-cutting
# ==========================================================================

def test_at_least_report_what_is_installed():
    """Never fails. Prints the adapter matrix so CI logs show coverage.

    Also prints the framework versions, because "the adapter tests passed" is
    only meaningful alongside what they passed against.
    """
    import importlib.metadata as md

    available = daftar.adapters.available()
    print("\nadapters available:", available or "none")
    for name in daftar.adapters.registry.all():
        mark = "yes" if name in available else "no"
        print(f"  {name:<12} {mark}")

    print("\nadapter status:")
    for name, (state, reason) in sorted(daftar.adapters.status().items()):
        detail = f"  {reason}" if reason else ""
        print(f"  {name:<12} {state}{detail}")

    print("\nframework versions:")
    for dist in ("jaxley", "jax", "jaxlib", "cpm-toolbox", "dm-meltingpot",
                 "dmlab2d", "numpy", "scipy"):
        try:
            print(f"  {dist:<14} {md.version(dist)}")
        except Exception:
            pass

    print("\nruntime checks (importing is not the same as working):")
    for name, probe in (("jaxley", _jaxley_is_usable), ("cpm", _cpm_is_usable)):
        if not daftar.adapters.get(name).is_available():
            continue
        ok, why = probe()
        print(f"  {name:<12} {'usable' if ok else 'UNUSABLE -- ' + why}")


# ==========================================================================
# Brian2
# ==========================================================================

def _brian2_network(name_suffix=""):
    """A small spiking network: neurons, random synapses, two monitors."""
    from brian2 import (
        Network, NeuronGroup, PopulationRateMonitor, SpikeMonitor, Synapses,
        ms, prefs, start_scope,
    )

    prefs.codegen.target = "numpy"     # no compiler needed in CI
    start_scope()
    eqs = """dv/dt = (I-v)/tau : 1
I : 1
tau : second"""
    G = NeuronGroup(50, eqs, threshold="v>1", reset="v=0", name="G")
    G.I = "1.5 + 0.5*rand()"
    G.tau = 10 * ms
    S = Synapses(G, G, on_pre="v_post += 0.05", name="S")
    S.connect(p=0.05)
    return Network(G, S, SpikeMonitor(G, name="spikes"),
                   PopulationRateMonitor(G, name="rate"))


@_require("brian2")
def test_brian2_records_the_resolved_integration_method(store):
    from brian2 import ms

    from daftar.adapters import brian2 as b2a

    with daftar.track("net", seed=42, store=store) as run:
        b2a.run_network(_brian2_network(), 50 * ms, run)
        rid = run.run_id

    m = store.load(rid)

    # The default is a candidate *list*; the winner is stored nowhere in Brian2.
    assert m.get("param.group.G.method_choice").startswith("(")
    assert m.get("param.group.G.method_resolved") == "exact"

    # codegen.target = auto resolves differently per machine; record the winner.
    assert m.get("param.brian2.codegen_resolved")
    assert m.get("param.brian2.prefs.core.default_float_dtype") == "float64"
    assert m.get("param.brian2.device") == "RuntimeDevice"

    # The schedule orders thresholds/synapses/resets within a timestep.
    assert "thresholds" in m.get("param.network.schedule")

    # Connectivity is drawn, not declared: p=0.05 gives a different graph
    # every unseeded run.
    assert int(m.get("param.synapses.S.n_synapses")) > 0
    assert m.get("param.group.G.N") == "50"
    assert m.get("param.group.G.equations_sha256")

    assert int(m.get("result.monitor.spikes.num_spikes")) > 0
    assert m.get("result.monitor.rate.mean_rate_hz")


@_require("brian2")
def test_brian2_seed_is_applied_so_connectivity_reproduces(store):
    """brian2.seed() is a device call; seeding numpy does not reach it.

    Without it, `connect(p=0.05)` draws a different graph on every run and two
    otherwise identical runs would differ with nothing to explain it.
    """
    from brian2 import ms

    from daftar.adapters import brian2 as b2a

    ids = []
    for _ in range(2):
        with daftar.track("net", seed=42, store=store) as run:
            b2a.run_network(_brian2_network(), 50 * ms, run)
            ids.append(run.run_id)

    a, b = store.load(ids[0]), store.load(ids[1])
    assert a.get("seed.brian2") == "true"
    assert a.get("param.synapses.S.n_synapses") == b.get("param.synapses.S.n_synapses")

    d = daftar.diff_manifests(a, b)
    assert not d.causes, [c.key for c in d.causes]
    assert not d.effects, [c.key for c in d.effects]


@_require("brian2")
def test_brian2_method_resolution_survives_brian_caching(store):
    """The resolved method must not vanish on the second run in a process.

    Brian2 logs its choice, but `apply_stateupdater` is cached, so the log line
    appears only the first time a set of equations is seen. Reading the log gave
    the method on run 1 and nothing on run 2, which showed up as a spurious
    cause in every diff. Resolution is now done directly.
    """
    from brian2 import ms

    from daftar.adapters import brian2 as b2a

    resolved = []
    for _ in range(2):
        with daftar.track("net", seed=1, store=store) as run:
            b2a.run_network(_brian2_network(), 20 * ms, run)
            rid = run.run_id
        resolved.append(store.load(rid).get("param.group.G.method_resolved"))

    assert resolved[0] == resolved[1] == "exact"
    assert "unknown" not in resolved


@_require("brian2")
def test_brian2_dt_change_is_a_cause_not_a_mystery(store):
    from brian2 import defaultclock, ms

    from daftar.adapters import brian2 as b2a

    with daftar.track("net", seed=42, store=store) as run:
        b2a.run_network(_brian2_network(), 50 * ms, run)
        a = run.run_id

    defaultclock.dt = 0.05 * ms
    try:
        with daftar.track("net", seed=42, store=store) as run:
            b2a.run_network(_brian2_network(), 50 * ms, run)
            b = run.run_id
    finally:
        defaultclock.dt = 0.1 * ms

    d = daftar.diff_manifests(store.load(a), store.load(b))
    causes = [c.key for c in d.causes]
    assert "param.brian2.defaultclock_dt" in causes
    assert d.effects, "changing dt should move the spike count"
