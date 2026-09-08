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

    if daftar.adapters.get("mne").is_available():
        from daftar.adapters.base import AVAILABLE, probe_import
        has_sklearn = probe_import("sklearn")[0] == AVAILABLE
        print(f"\nmne optional deps: scikit-learn "
              f"{'present (ICA method=fastica)' if has_sklearn else 'MISSING (ICA falls back to infomax)'}")

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


# ==========================================================================
# MNE-Python
# ==========================================================================

def _mne_ica_method() -> str:
    """An ICA method that works in this environment.

    MNE's default, `fastica`, delegates to scikit-learn, which is an *optional*
    MNE dependency -- so `import mne` succeeds and `ica.fit()` raises
    ImportError. `infomax` is implemented natively in MNE and needs nothing
    extra.

    Choosing rather than skipping is deliberate: what these tests exercise is
    the adapter's recording of exclusions, convergence and random_state, and
    none of that depends on which algorithm ran. Skipping would lose real
    coverage over an incidental dependency.
    """
    from daftar.adapters.base import AVAILABLE, probe_import

    return "fastica" if probe_import("sklearn")[0] == AVAILABLE else "infomax"


def _mne_raw(n_channels=6, sfreq=250.0, seconds=12, seed=0):
    """A small synthetic EEG recording with one channel marked bad."""
    import numpy as np
    import mne

    mne.set_log_level("ERROR")
    names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(names, sfreq, "eeg")
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=2e-5, size=(n_channels, int(sfreq * seconds)))
    raw = mne.io.RawArray(data, info)
    raw.info["bads"] = [names[2]]
    return raw


@_require("mne")
def test_mne_records_recording_state_without_subject_data(store):
    """Structure is recorded; subject identifiers are not."""
    import mne

    from daftar.adapters import mne as mnea

    raw = _mne_raw()
    raw.info["subject_info"] = {"his_id": "PATIENT-12345", "sex": 1}
    raw.info["line_freq"] = 50.0

    with daftar.track("preproc", seed=42, store=store) as run:
        mnea.describe_environment(run)
        mnea.describe_raw(raw, run)
        rid = run.run_id

    m = store.load(rid)

    assert m.get("param.recording.sfreq") == "250.0"
    assert m.get("param.recording.nchan") == "6"
    assert m.get("param.recording.n_eeg") == "6"
    assert m.get("param.recording.line_freq") == "50.0"

    # Bad channels are a human judgement that changes everything downstream.
    assert "EEG002" in m.get("param.recording.bads")
    assert m.get("param.recording.n_bads") == "1"

    # Subject data is hashed, never stored. Manifests get committed.
    assert m.get("param.recording.subject_info_sha256")
    assert "PATIENT-12345" not in m.to_json()
    assert m.get("param.recording.meas_date_set") in ("true", "false")
    assert mne.__version__ == m.get("param.mne.version")


@_require("mne")
def test_mne_filter_records_the_design_not_just_the_band(store):
    """info['highpass']/['lowpass'] survive raw.filter(); the design does not.

    A zero-phase FIR with a wide transition band and a causal IIR are different
    filters. "1-40 Hz" in a methods section does not distinguish them.
    """
    from daftar.adapters import mne as mnea

    raw = _mne_raw()
    with daftar.track("filt", seed=42, store=store) as run:
        mnea.filter_raw(raw, run, l_freq=1.0, h_freq=40.0, fir_design="firwin")
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.filter.l_freq") == "1.0"
    assert m.get("param.filter.h_freq") == "40.0"
    assert m.get("param.filter.fir_design") == "firwin"
    # Defaults the caller never passed are recorded and flagged as defaults, so
    # a future change of MNE default shows up as a diff.
    assert m.get("param.filter.phase") == "zero"
    assert m.get("param.filter.phase.was_default") == "true"
    assert m.get("param.filter.fir_design.was_default") is None
    # The resulting band is refreshed from Info afterwards.
    assert m.get("param.recording.highpass") == "1.0"


@_require("mne")
def test_mne_epochs_record_what_was_thrown_away(store):
    """An average over 40 surviving epochs differs from one over 180."""
    import numpy as np
    import mne

    from daftar.adapters import mne as mnea

    raw = _mne_raw(seconds=12)
    events = np.array([[int(250 * t), 0, 1] for t in range(1, 11)])

    with daftar.track("epoch", seed=42, store=store) as run:
        epochs = mne.Epochs(raw, events, event_id={"stim": 1},
                            tmin=-0.2, tmax=0.5, baseline=(None, 0),
                            reject=dict(eeg=1e-6),   # aggressive: drops most
                            preload=True)
        mnea.describe_epochs(epochs, run)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.epochs.tmin") == "-0.2"
    assert m.get("param.epochs.tmax") == "0.5"
    assert "eeg" in m.get("param.epochs.reject")
    assert m.get("param.epochs.n_conditions") == "1"

    total = int(m.get("result.epochs.n_epochs_total"))
    dropped = int(m.get("result.epochs.n_epochs_dropped"))
    assert total == 10
    assert dropped > 0, "the aggressive reject threshold should drop epochs"
    assert m.get("result.epochs.drop_rate")
    # Why they were dropped, not just how many.
    assert m.get("result.epochs.drop_reasons")


@_require("mne")
def test_mne_ica_records_exclusions_and_convergence(store):
    """ica.exclude is the most consequential unrecorded decision in EEG."""
    import mne

    from daftar.adapters import mne as mnea

    raw = _mne_raw(n_channels=6)
    method = _mne_ica_method()
    ica = mne.preprocessing.ICA(n_components=4, method=method,
                                random_state=97, max_iter=200)
    ica.fit(raw)
    ica.exclude = [0, 2]

    with daftar.track("ica", seed=42, store=store) as run:
        mnea.apply_ica(ica, raw.copy(), run)
        rid = run.run_id

    m = store.load(rid)

    assert m.get("param.ica.method") == method
    assert m.get("param.ica.random_state") == "97"
    assert m.get("param.ica.n_components_fitted") == "4"

    # The human decision.
    assert m.get("param.ica.exclude") == "[0, 2]"
    assert m.get("param.ica.n_excluded") == "2"

    # n_iter_ == max_iter means it stopped, not that it finished.
    assert m.get("result.ica.n_iter")
    assert m.get("result.ica.converged") in ("true", "false")


@_require("mne")
def test_mne_unseeded_ica_is_flagged_as_irreproducible(store):
    """random_state=None means the components -- and the exclusions indexing
    into them -- differ between runs. The manifest must say so loudly."""
    import mne

    from daftar.adapters import mne as mnea

    ica = mne.preprocessing.ICA(n_components=3, method=_mne_ica_method(),
                                random_state=None, max_iter=100)
    ica.fit(_mne_raw(n_channels=5))

    with daftar.track("ica-unseeded", store=store) as run:
        mnea.describe_ica(ica, run)
        rid = run.run_id

    assert "NOT REPRODUCIBLE" in store.load(rid).get("param.ica.random_state")


@_require("mne")
def test_mne_bad_channel_change_is_a_cause(store):
    """Marking one more channel bad changes every downstream number."""
    from daftar.adapters import mne as mnea

    ids = []
    for bads in (["EEG002"], ["EEG002", "EEG004"]):
        raw = _mne_raw()
        raw.info["bads"] = bads
        with daftar.track("preproc", seed=42, store=store) as run:
            mnea.describe_raw(raw, run)
            ids.append(run.run_id)

    causes = [c.key for c in
              daftar.diff_manifests(store.load(ids[0]), store.load(ids[1])).causes]
    assert "param.recording.bads" in causes
    assert "param.recording.n_bads" in causes


# ==========================================================================
# sbi
# ==========================================================================

def _sbi_setup(n_sims=200, dim=2, seed=0):
    """A tiny amortised inference problem: linear Gaussian simulator."""
    import torch
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    prior = BoxUniform(low=-2 * torch.ones(dim), high=2 * torch.ones(dim))

    def simulator(theta):
        return theta + 0.1 * torch.randn_like(theta)

    theta = prior.sample((n_sims,))
    return prior, theta, simulator(theta)


@_require("sbi")
def test_sbi_records_training_hyperparameters_sbi_discards(store):
    """train() configures the fit and is then thrown away by sbi."""
    from sbi.inference import NPE

    from daftar.adapters import sbi as sbia

    prior, theta, x = _sbi_setup()
    inference = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)

    with daftar.track("npe", seed=42, store=store) as run:
        sbia.append_simulations(inference, theta, x, run)
        sbia.train(inference, run, training_batch_size=50, max_num_epochs=5,
                   learning_rate=1e-3)
        rid = run.run_id

    m = store.load(rid)

    # None of these survive on the trainer after the call.
    assert m.get("param.training.training_batch_size") == "50"
    assert m.get("param.training.learning_rate") == "0.001"
    assert m.get("param.training.max_num_epochs") == "5"
    # Defaults never passed are recorded and flagged, so an sbi default change
    # shows up as a diff rather than silently moving the result.
    assert m.get("param.training.stop_after_epochs") == "20"
    assert m.get("param.training.stop_after_epochs.was_default") == "true"
    assert m.get("param.training.learning_rate.was_default") is None

    # sbi's `NPE` is an alias; the concrete class is NPE_C. Recording what
    # actually ran is the point -- an alias can be repointed at a different
    # algorithm in a later release without the calling code changing.
    assert m.get("param.sbi.method") == "NPE_C"
    assert m.get("param.sbi.num_simulations_total") == "200"


@_require("sbi")
def test_sbi_flags_training_that_hit_the_epoch_limit(store):
    """Stopping on plateau and stopping on max_num_epochs are different outcomes.

    sbi raises a UserWarning for the second and stores nothing you would notice,
    so a posterior from a truncated fit looks like any other.
    """
    from sbi.inference import NPE

    from daftar.adapters import sbi as sbia

    prior, theta, x = _sbi_setup()
    inference = NPE(prior=prior, show_progress_bars=False)

    with daftar.track("npe-truncated", seed=42, store=store) as run:
        sbia.append_simulations(inference, theta, x, run)
        sbia.train(inference, run, training_batch_size=50, max_num_epochs=3,
                   stop_after_epochs=1000)   # guarantees the limit is what stops it
        rid = run.run_id

    m = store.load(rid)
    assert m.get("result.training.converged") == "false"
    assert int(m.get("result.training.epochs_trained_last")) >= 3
    assert m.get("result.training.best_validation_loss_last")


@_require("sbi")
def test_sbi_records_the_resolved_architecture_not_the_string(store):
    """density_estimator="maf" becomes a flow whose depth and width are defaults."""
    from sbi.inference import NPE

    from daftar.adapters import sbi as sbia

    prior, theta, x = _sbi_setup()
    inference = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)

    with daftar.track("npe-arch", seed=42, store=store) as run:
        sbia.append_simulations(inference, theta, x, run)
        sbia.train(inference, run, training_batch_size=50, max_num_epochs=3)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.estimator.class")
    assert int(m.get("param.estimator.n_parameters")) > 0
    assert m.get("param.estimator.input_shape") == "[2]"
    assert m.get("param.estimator.condition_shape") == "[2]"

    # The prior is an experimental parameter, not scaffolding.
    assert m.get("param.prior.type") == "BoxUniform"
    assert m.get("param.prior.n_dims") == "2"
    assert "-2.0" in m.get("param.prior.low")


@_require("sbi")
def test_sbi_records_the_proposal_per_round(store):
    """In sequential methods the proposal is the algorithm.

    Round 1 draws from the prior, later rounds from the current posterior. sbi
    records how many rounds happened but not what each drew from, so an
    amortised run and a sequential run of the same budget look alike.
    """
    from sbi.inference import NPE

    from daftar.adapters import sbi as sbia

    prior, theta, x = _sbi_setup(n_sims=150)
    inference = NPE(prior=prior, show_progress_bars=False)

    with daftar.track("snpe", seed=42, store=store) as run:
        sbia.append_simulations(inference, theta, x, run)
        estimator = sbia.train(inference, run, training_batch_size=50,
                               max_num_epochs=3)
        posterior = inference.build_posterior(estimator)

        import torch
        x_o = torch.zeros(1, 2)
        posterior.set_default_x(x_o)
        theta2 = posterior.sample((100,), show_progress_bars=False)
        x2 = theta2 + 0.1 * torch.randn_like(theta2)
        sbia.append_simulations(inference, theta2, x2, run, proposal=posterior)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.sbi.round_0.proposal") == "prior"
    assert m.get("param.sbi.round_1.proposal") == "DirectPosterior"
    assert m.get("param.sbi.round_0.n_simulations") == "150"
    assert m.get("param.sbi.round_1.n_simulations") == "100"
    # Content hashes, so a changed simulation batch is detectable.
    assert len(m.get("param.sbi.round_0.theta_sha256")) == 16


@_require("sbi")
def test_sbi_posterior_records_the_observation_by_hash(store):
    """x_o is hashed, not stored: it can be large and is often measured data."""
    import torch
    from sbi.inference import NPE

    from daftar.adapters import sbi as sbia

    prior, theta, x = _sbi_setup()
    inference = NPE(prior=prior, show_progress_bars=False)

    ids = []
    for x_o in (torch.zeros(1, 2), torch.ones(1, 2)):
        inf = NPE(prior=prior, show_progress_bars=False)
        with daftar.track("npe-post", seed=42, store=store) as run:
            sbia.append_simulations(inf, theta, x, run)
            est = sbia.train(inf, run, training_batch_size=50, max_num_epochs=3)
            posterior = inf.build_posterior(est)
            sbia.sample_posterior(posterior, (50,), run, x=x_o,
                                  show_progress_bars=False)
            ids.append(run.run_id)

    a, b = store.load(ids[0]), store.load(ids[1])
    assert len(a.get("param.posterior.x_o_sha256")) == 16
    # A different observation is a cause of a different posterior.
    causes = [c.key for c in daftar.diff_manifests(a, b).causes]
    assert "param.posterior.x_o_sha256" in causes
    assert a.get("result.posterior.mean")


# ==========================================================================
# Nilearn
# ==========================================================================

def _nilearn_data(n_vols=40, seed=0):
    """A tiny synthetic 4D image plus an fMRIPrep-shaped confounds table."""
    import nibabel as nib
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    img = nib.Nifti1Image(
        rng.normal(size=(6, 6, 6, n_vols)).astype("float32"), np.eye(4)
    )
    confounds = pd.DataFrame(
        rng.normal(size=(n_vols, 8)),
        columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                 "csf", "white_matter"],
    )
    return img, confounds


@_require("nilearn")
def test_nilearn_records_confounds_that_the_masker_forgets(store):
    """fit_transform(confounds=...) regresses them out and stores nothing.

    Which columns were regressed is one of the largest free parameters in fMRI
    and is not recoverable from the masker afterwards -- get_params() has no
    `confounds` key. This is the ica.exclude of fMRI.
    """
    from nilearn.maskers import NiftiMasker

    from daftar.adapters import nilearn as nla

    img, confounds = _nilearn_data()
    masker = NiftiMasker(standardize="zscore_sample", detrend=True)

    with daftar.track("denoise", seed=42, store=store) as run:
        nla.describe_environment(run)
        nla.describe_image(img, run)
        nla.fit_transform(masker, img, run, confounds=confounds)
        rid = run.run_id

    m = store.load(rid)

    assert m.get("param.confounds.applied") == "true"
    assert m.get("param.confounds.n_regressors") == "8"
    # The taxonomy, so a diff can say "motion changed" not "compare two lists".
    assert m.get("param.confounds.n_motion") == "6"
    assert m.get("param.confounds.n_tissue") == "2"
    assert "trans_x" in m.get("param.confounds.names")
    # Values are subject data: hashed, never stored.
    assert len(m.get("param.confounds.values_sha256")) == 16
    assert "0." not in m.get("param.confounds.names")

    # The declared config is recorded too.
    assert m.get("param.masker.standardize") == "zscore_sample"
    assert m.get("param.masker.detrend") == "true"


@_require("nilearn")
def test_nilearn_records_the_mask_that_actually_resolved(store):
    """mask_img=None means the mask is computed from the data, per subject."""
    from nilearn.maskers import NiftiMasker

    from daftar.adapters import nilearn as nla

    img, _ = _nilearn_data()
    masker = NiftiMasker(standardize="zscore_sample")

    with daftar.track("mask", seed=42, store=store) as run:
        nla.fit_transform(masker, img, run)
        rid = run.run_id

    m = store.load(rid)
    # Declared: None. Resolved: a specific number of voxels.
    assert m.get("param.masker.mask_img") == "null"
    assert m.get("param.masker.mask_resolved") == "true"
    assert int(m.get("param.masker.mask_n_voxels")) > 0
    assert int(m.get("result.masker.timeseries.n_timepoints")) == 40
    assert m.get("param.confounds.applied") == "false"


@_require("nilearn")
def test_nilearn_changed_confound_set_is_a_cause(store):
    """Dropping the tissue regressors changes every downstream number."""
    from nilearn.maskers import NiftiMasker

    from daftar.adapters import nilearn as nla

    img, confounds = _nilearn_data()
    ids = []
    for cols in (list(confounds.columns), ["trans_x", "trans_y", "trans_z"]):
        with daftar.track("denoise", seed=42, store=store) as run:
            nla.fit_transform(NiftiMasker(standardize="zscore_sample"), img, run,
                              confounds=confounds[cols])
            ids.append(run.run_id)

    a, b = store.load(ids[0]), store.load(ids[1])
    causes = [c.key for c in daftar.diff_manifests(a, b).causes]
    assert "param.confounds.names" in causes
    assert "param.confounds.n_regressors" in causes
    assert "param.confounds.values_sha256" in causes


@_require("nilearn")
def test_nilearn_records_parcellation_and_connectivity_kind(store):
    """`kind` changes the numbers entirely; n_labels identifies the atlas."""
    import nibabel as nib
    import numpy as np
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker

    from daftar.adapters import nilearn as nla

    img, confounds = _nilearn_data()
    rng = np.random.default_rng(1)
    atlas = nib.Nifti1Image(
        rng.integers(0, 5, size=(6, 6, 6)).astype("int16"), np.eye(4)
    )

    with daftar.track("connectivity", seed=42, store=store) as run:
        masker = NiftiLabelsMasker(atlas, standardize="zscore_sample")
        ts = nla.fit_transform(masker, img, run, confounds=confounds)
        measure = ConnectivityMeasure(kind="correlation", vectorize=True)
        nla.connectivity_fit_transform(measure, [ts], run)
        rid = run.run_id

    m = store.load(rid)
    assert int(m.get("param.masker.n_labels")) >= 4
    assert m.get("param.masker.labels_sha256")
    assert m.get("param.connectivity.kind") == "correlation"
    assert m.get("param.connectivity.vectorize") == "true"
    assert m.get("result.connectivity.shape")

    # cov_estimator is declared None and resolves to Ledoit-Wolf shrinkage,
    # which pulls the covariance toward the identity and can dominate the
    # result. get_params() reports only the None.
    assert m.get("param.connectivity.cov_estimator") == "null"
    assert m.get("param.connectivity.cov_estimator_resolved") == "LedoitWolf"
    assert m.get("param.connectivity.cov_estimator.was_default") == "true"


@_require("nilearn")
def test_nilearn_glm_records_the_design_matrix(store):
    """The design matrix is the model; its columns encode every choice."""
    import numpy as np
    import pandas as pd
    from nilearn.glm.first_level import FirstLevelModel

    from daftar.adapters import nilearn as nla

    import nibabel as nib

    img, _ = _nilearn_data(n_vols=40)
    # An explicit mask: nilearn's automatic masking finds nothing in pure noise,
    # and a real analysis supplies a brain mask anyway.
    mask = nib.Nifti1Image(np.ones((6, 6, 6), dtype="uint8"), np.eye(4))
    events = pd.DataFrame({
        "onset": [4.0, 20.0, 44.0],
        "duration": [4.0, 4.0, 4.0],
        "trial_type": ["a", "b", "a"],
    })

    with daftar.track("glm", seed=42, store=store) as run:
        model = FirstLevelModel(t_r=2.0, hrf_model="spm", drift_model="cosine",
                                high_pass=0.01, noise_model="ar1",
                                mask_img=mask, minimize_memory=False)
        model.fit(img, events=events)
        nla.describe_glm(model, run)
        nla.describe_contrast("a_minus_b", model.compute_contrast("a - b"), run)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.glm.t_r") == "2.0"
    assert m.get("param.glm.hrf_model") == "spm"
    assert m.get("param.glm.noise_model") == "ar1"
    assert m.get("param.glm.n_design_columns")
    assert m.get("param.glm.design_sha256")
    assert "a" in m.get("param.glm.design_columns")
    assert m.get("result.contrast.a_minus_b.max")
