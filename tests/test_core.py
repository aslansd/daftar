import json
import os
import zipfile
from pathlib import Path

import pytest

import daftar
from daftar.diff import (
    EXPLAINED, IDENTICAL, NO_EFFECT, NONDETERMINISTIC, diff_manifests, render_diff,
)
from daftar.manifest import Manifest
from daftar.store import RunStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated store in an isolated working directory.

    The chdir matters. Without it, every test inherits the git state of
    whatever directory pytest was launched from, so a dirty working tree in the
    developer's checkout silently changes what the tests assert. That is how
    test_clean_run_has_no_blockers passed on a machine where the project was
    not a git repository and failed on one where it was.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAFTAR_DIR", str(tmp_path / ".daftar"))
    s = RunStore()
    s.init()
    return s


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A clean, fully committed git repository as the working directory."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "T")
    (repo / "sim.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-m", "init")

    monkeypatch.chdir(repo)
    monkeypatch.setenv("DAFTAR_DIR", str(repo / ".daftar"))
    return repo


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def test_manifest_roundtrip():
    m = Manifest(run_id="r-1")
    m.set("param.dt", 0.025)
    m.set("param.solver", "bwd_euler")
    m.set("result.rate", 4.8)
    back = Manifest.from_json(m.to_json())
    assert back.run_id == "r-1"
    assert back.fields == m.fields


def test_manifest_values_are_stable_strings():
    m = Manifest(run_id="r-1")
    m.set("param.a", 0.1)
    m.set("param.b", True)
    m.set("param.c", [1, 2, 3])
    m.set("param.d", None)
    assert m.fields["param.a"] == "0.1"
    assert m.fields["param.b"] == "true"
    assert m.fields["param.c"] == "[1, 2, 3]"
    assert m.fields["param.d"] == "null"


def test_manifest_key_ordering_groups_namespaces():
    m = Manifest(run_id="r-1")
    for k in ("result.x", "param.b", "meta.label", "env.python", "param.a"):
        m.set(k, "v")
    ordered = m.ordered_keys
    assert ordered.index("meta.label") < ordered.index("param.a")
    assert ordered.index("param.a") < ordered.index("param.b")
    assert ordered.index("param.b") < ordered.index("env.python")
    assert ordered.index("env.python") < ordered.index("result.x")


def test_rejects_unknown_schema():
    with pytest.raises(ValueError, match="schema"):
        Manifest.from_dict({"schema": "99", "run_id": "r", "fields": {}})


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------

def test_track_records_and_persists(store):
    with daftar.track("demo", params={"dt": 0.025}, seed=42) as run:
        run.log_result("rate", 4.8)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("meta.status") == "completed"
    assert m.get("param.dt") == "0.025"
    assert m.get("result.rate") == "4.8"
    assert m.get("seed.value") == "42"
    assert m.get("seed.was_explicit") == "true"
    assert m.get("env.python")
    assert float(m.get("cost.wall_clock_s")) >= 0


def test_seed_is_actually_applied(store):
    import random
    with daftar.track("a", seed=123):
        first = [random.random() for _ in range(3)]
    with daftar.track("b", seed=123):
        second = [random.random() for _ in range(3)]
    assert first == second, "track() must set seeds, not merely record them"


def test_unseeded_run_still_records_a_usable_seed(store):
    with daftar.track("noseed") as run:
        rid, seed = run.run_id, run.seed
    m = store.load(rid)
    assert m.get("seed.was_explicit") == "false"
    assert m.get("seed.value") == str(seed)


def test_failed_run_is_recorded_not_lost(store):
    with pytest.raises(ValueError):
        with daftar.track("boom", params={"x": 1}) as run:
            rid = run.run_id
            raise ValueError("solver diverged")

    m = store.load(rid)
    assert m.get("meta.status") == "failed"
    assert m.get("meta.error_type") == "ValueError"
    assert "diverged" in m.get("meta.error")
    assert m.get("param.x") == "1"


def test_decorator_captures_defaults_and_results(store):
    @daftar.tracked(seed=7)
    def simulate(dt=0.025, solver="bwd_euler"):
        return {"rate": 4.8, "trace": [1, 2, 3]}

    simulate()
    m = store.load(simulate.last_run_id)
    # The default the caller never passed is the one they forget six weeks later.
    assert m.get("param.dt") == "0.025"
    assert m.get("param.solver") == "bwd_euler"
    assert m.get("result.rate") == "4.8"
    assert m.get("result.trace") is None, "non-scalars must not pollute results"


def test_input_and_output_hashing(tmp_path, store):
    data = tmp_path / "in.csv"
    data.write_text("a,b\n1,2\n")
    out = tmp_path / "fig.txt"

    with daftar.track("io") as run:
        run.add_input(data)
        out.write_text("result")
        run.add_output(out)
        rid = run.run_id

    m = store.load(rid)
    assert len(m.get("input.in.csv.sha256")) == 12
    assert m.get("input.in.csv.bytes") == "8"
    assert m.get("output.fig.txt.sha256") != "missing"


def test_missing_output_does_not_lose_the_run(tmp_path, store):
    with daftar.track("io") as run:
        run.add_output(tmp_path / "never_written.png")
        rid = run.run_id
    assert store.load(rid).get("output.never_written.png.sha256") == "missing"


def test_nested_params_are_flattened(store):
    with daftar.track("nested", params={"solver": {"name": "bwd", "rtol": 1e-6}}) as run:
        rid = run.run_id
    m = store.load(rid)
    assert m.get("param.solver.name") == "bwd"
    assert m.get("param.solver.rtol") == "1e-06"


# --------------------------------------------------------------------------
# diff -- the reason anyone installs this
# --------------------------------------------------------------------------

def _mk(run_id, **fields):
    m = Manifest(run_id=run_id)
    for k, v in fields.items():
        m.set(k.replace("__", "."), v)
    return m


def test_identical_runs():
    a = _mk("r-a", param__dt=0.025, result__rate=4.8)
    b = _mk("r-b", param__dt=0.025, result__rate=4.8)
    d = diff_manifests(a, b)
    assert d.verdict == IDENTICAL
    assert d.is_reproduction


def test_explained_difference_separates_cause_from_effect():
    a = _mk("r-a", env__jax="0.4.35", param__dt=0.025, result__rate=4.812)
    b = _mk("r-b", env__jax="0.4.41", param__dt=0.025, result__rate=4.796)
    d = diff_manifests(a, b)
    assert d.verdict == EXPLAINED
    assert [c.key for c in d.causes] == ["env.jax"]
    assert [c.key for c in d.effects] == ["result.rate"]


def test_nondeterminism_is_detected_and_named():
    """The finding that matters most: nothing changed but the answer did."""
    a = _mk("r-a", param__dt=0.025, seed__value=1, result__rate=4.812)
    b = _mk("r-b", param__dt=0.025, seed__value=1, result__rate=4.400)
    d = diff_manifests(a, b)
    assert d.verdict == NONDETERMINISTIC
    assert not d.causes
    assert "not deterministic" in d.verdict_text


def test_change_without_effect_is_evidence_of_robustness():
    a = _mk("r-a", env__numpy="1.26.0", result__rate=4.8)
    b = _mk("r-b", env__numpy="2.4.4", result__rate=4.8)
    d = diff_manifests(a, b)
    assert d.verdict == NO_EFFECT


def test_timestamps_never_count_as_differences():
    a = _mk("r-a", meta__started_at="2026-01-01T00:00:00", param__dt=0.025)
    b = _mk("r-b", meta__started_at="2026-06-01T00:00:00", param__dt=0.025)
    d = diff_manifests(a, b)
    assert d.verdict == IDENTICAL, "clock drift must not read as a difference"


def test_cost_is_neutral_not_an_effect():
    a = _mk("r-a", param__dt=0.025, cost__wall_clock_s=318, result__rate=4.8)
    b = _mk("r-b", param__dt=0.025, cost__wall_clock_s=305, result__rate=4.8)
    d = diff_manifests(a, b)
    assert d.verdict == IDENTICAL
    assert len(d.neutral) == 1


def test_render_diff_is_readable():
    a = _mk("r-a", env__jax="0.4.35", result__rate=4.812)
    b = _mk("r-b", env__jax="0.4.41", result__rate=4.796)
    text = render_diff(diff_manifests(a, b))
    assert "candidate causes (1)" in text
    assert "observed effects (1)" in text
    assert "0.4.35" in text and "0.4.41" in text


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def test_grid_is_a_cartesian_product():
    combos = list(daftar.grid(dt=[0.025, 0.01], solver=["a", "b"]))
    assert len(combos) == 4
    assert {"dt": 0.025, "solver": "a"} in combos


def test_sweep_records_each_point_separately(store):
    def sim(dt, solver):
        return {"rate": 1.0 / dt}

    result = daftar.sweep(sim, dt=[0.025, 0.01], solver=["bwd_euler"], store=store)
    assert len(result.run_ids) == 2
    assert not result.failed
    rows = result.table(store=store)
    assert len(rows) == 2
    assert "param.dt" in rows[0]


def test_sweep_survives_a_failing_point(store):
    def sim(dt):
        if dt == 0.01:
            raise RuntimeError("diverged")
        return {"rate": 1.0}

    result = daftar.sweep(sim, dt=[0.025, 0.01, 0.05], store=store)
    assert len(result.run_ids) == 3
    assert len(result.failed) == 1
    statuses = [store.load(r).get("meta.status") for r in result.run_ids]
    assert statuses.count("completed") == 2
    assert statuses.count("failed") == 1


def test_compare_many_finds_the_swept_axes(store):
    daftar.sweep(lambda dt: {"rate": 1 / dt}, dt=[0.025, 0.01], store=store)
    varying = daftar.compare_many(store.list())
    assert "param.dt" in varying
    assert varying["param.dt"] == {"0.025", "0.01"}


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def test_replay_plan_flags_a_changed_input(tmp_path, store):
    data = tmp_path / "in.csv"
    data.write_text("original")
    with daftar.track("r") as run:
        run.add_input(data)
        rid = run.run_id

    data.write_text("tampered")
    plan = daftar.plan_replay(store.load(rid))
    assert not plan.reproducible
    assert any("changed since the run" in b for b in plan.blockers)


def test_replay_plan_flags_a_missing_input(tmp_path, store):
    data = tmp_path / "in.csv"
    data.write_text("x")
    with daftar.track("r") as run:
        run.add_input(data)
        rid = run.run_id
    data.unlink()
    plan = daftar.plan_replay(store.load(rid))
    assert any("missing" in b for b in plan.blockers)


def test_clean_run_has_no_blockers(git_repo):
    """A committed working tree must produce a reproducible plan."""
    store = RunStore()
    with daftar.track("clean", params={"dt": 0.025}, seed=1) as run:
        rid = run.run_id
    plan = daftar.plan_replay(store.load(rid))
    assert plan.reproducible, plan.blockers
    assert "0.025" in plan.render()


def test_dirty_tree_blocks_replay(git_repo):
    """The mirror image, and the behaviour that surfaced this bug.

    Uncommitted edits are not recoverable from a manifest, so a run made on a
    dirty tree is honestly reported as unreproducible rather than optimistically
    marked fine.
    """
    store = RunStore()
    (git_repo / "sim.py").write_text("x = 2  # uncommitted\n")
    with daftar.track("dirty", seed=1) as run:
        rid = run.run_id
    plan = daftar.plan_replay(store.load(rid))
    assert not plan.reproducible
    assert any("dirty" in b for b in plan.blockers)


def test_entrypoint_points_at_the_caller(store):
    """Frame walking must land on the caller, not on the test runner.

    A fixed frame depth put `_pytest/python.py` in the manifest; the depth
    depends on who is calling, so it is found by search instead.
    """
    with daftar.track("entry") as run:
        rid = run.run_id
    entry = store.load(rid).get("code.entrypoint")
    file_part = entry.split("::")[0]
    assert file_part.endswith("test_core.py"), entry
    assert "site-packages" not in file_part, entry


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_bundle_is_self_describing(tmp_path, store):
    data = tmp_path / "in.csv"
    data.write_text("a,b\n1,2\n")
    with daftar.track("exp", params={"dt": 0.025}, seed=3) as run:
        run.add_input(data)
        run.log_result("rate", 4.8)
        rid = run.run_id

    out = daftar.export_bundle(store.load(rid), tmp_path / "bundle.zip", store=store)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "README.md" in names
        assert "manifest.json" in names
        assert "fields.tsv" in names
        assert "inputs/in.csv" in names
        readme = zf.read("README.md").decode()
        # Readable without the tool installed.
        assert "dt" in readme and "0.025" in readme
        assert "rate" in readme
    assert daftar.load_bundle(out).run_id == rid


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def test_short_id_prefix_resolves(store):
    with daftar.track("x") as run:
        rid = run.run_id
    assert store.load(rid[:6]).run_id == rid


def test_reindex_rebuilds_from_manifests(store):
    for _ in range(3):
        with daftar.track("x"):
            pass
    store.db_path.unlink()
    store._conn = None
    assert store.reindex() == 3


def test_manifests_are_valid_json_on_disk(store):
    with daftar.track("x", params={"a": 1}) as run:
        rid = run.run_id
    raw = (store.runs_dir / f"{rid}.json").read_text()
    parsed = json.loads(raw)
    assert parsed["run_id"] == rid
    assert parsed["fields"]["param.a"] == "1"


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------

def test_adapters_import_without_their_frameworks():
    from daftar import adapters
    assert set(adapters.registry.all()) == {"jaxley", "cpm", "meltingpot"}
    for name in adapters.registry.all():
        assert isinstance(adapters.get(name).is_available(), bool)


def test_safe_swallows_probe_failures():
    from daftar.adapters.base import safe

    def boom():
        raise AttributeError("framework renamed this")

    assert safe(boom, default="<unavailable>") == "<unavailable>"


# --------------------------------------------------------------------------
# the observer-effect regression
# --------------------------------------------------------------------------

def test_daftar_store_does_not_make_the_repo_look_dirty(git_repo):
    """Using daftar must not itself register as a change to the experiment.

    Before this was fixed, the first run in a clean repo was recorded clean and
    every subsequent run was dirty -- because .daftar/runs/ now existed as an
    untracked path. Every diff then showed a spurious code.dirty change.
    """
    store = RunStore()
    # Same seed, so the only thing that could differ is the git state.
    with daftar.track("first", seed=1, store=store) as r1:
        first = r1.run_id
    with daftar.track("second", seed=1, store=store) as r2:
        second = r2.run_id

    a, b = store.load(first), store.load(second)
    assert a.get("code.dirty") == "false"
    assert b.get("code.dirty") == "false", "the store must not dirty the tree"
    assert not diff_manifests(a, b).causes


def test_unseeded_runs_differ_by_seed_and_that_is_a_real_cause(store):
    """Two unseeded runs are genuinely different experiments. Say so.

    The assertion is deliberately about the *namespace*, not an exact field
    list. How many `seed.*` fields exist depends on which RNG libraries happen
    to be imported: with jax present, `apply_seeds` also records
    `seed.jax_root_key`, which is derived from the seed and therefore differs
    too. An earlier version asserted `== ["seed.value"]` and passed only in
    environments without jax -- the same environment-dependence this package
    exists to expose.
    """
    with daftar.track("a", store=store) as r1:
        first = r1.run_id
    with daftar.track("b", store=store) as r2:
        second = r2.run_id

    causes = diff_manifests(store.load(first), store.load(second)).causes
    keys = [c.key for c in causes]

    assert "seed.value" in keys
    assert all(k.startswith("seed.") for k in keys), (
        f"only the seed should differ between two unseeded runs, got {keys}"
    )


def test_install_source_is_recorded_for_non_index_installs(store):
    """A version string is not an identity.

    Two builds can report the same version and behave differently -- the case
    that motivated this: jaxley 0.13.0 on PyPI is broken with current JAX, and
    jaxley `main` fixes it while still calling itself 0.13.0. A manifest that
    recorded only `env.jaxley = 0.13.0` would call those two environments
    identical when one works and one does not.

    PEP 610 records the true origin in direct_url.json for anything not
    installed from an index. daftar itself is installed editable during
    development, so it is a reliable subject here.
    """
    from daftar import capture

    env = capture.environment()
    assert env.get("daftar"), "daftar should record its own version"
    source = env.get("daftar.source")
    if source is not None:  # None when installed normally from PyPI
        assert source.startswith(("editable:", "git+", "local:")), source


def test_install_source_absent_for_stdlib(store):
    """No direct_url.json means an index install; absence is the signal."""
    from daftar import capture

    assert capture._install_source("definitely-not-a-real-package") is None


def test_cpm_adapter_reads_cohort_from_the_optimiser_not_the_model(store):
    """The cpm Wrapper holds one participant; the optimiser holds the cohort.

    cpm calls `model.reset(data=participant)` per subject, so a Wrapper's
    `.data` is a single-participant template. Reading it reports a cohort of 1
    however many people are in the study. Confidently wrong provenance is worse
    than none, so this is pinned with a stub rather than left to the live tests,
    which only run when cpm is installed.
    """
    pd = pytest.importorskip("pandas")
    import numpy as np

    from daftar.adapters import cpm as cpma

    full = pd.DataFrame({
        "ppt": [1, 1, 1, 2, 2, 2],
        "stimulus": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "observed": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
    })
    grouped = full.groupby("ppt")
    groups = list(grouped.groups.keys())

    class WrapperStub:
        data = grouped.get_group(groups[0])   # one participant -- the trap
        parameters = None

    class OptimiserStub:
        model = WrapperStub()
        data = grouped                         # the real cohort
        participants = WrapperStub.data        # cpm's misleading name
        initial_guess = np.zeros((2, 1))
        __parallel__ = False
        __libraries__ = ["numpy"]
        cl = None
        ppt_identifier = "ppt"
        prior = False
        kwargs = {"approx_grad": True}
        loss = type("f", (), {"__name__": "continuous"})()
        fit: list = []
        parameters: list = []

    stub = OptimiserStub()
    stub.groups = groups

    with daftar.track("cohort", store=store) as run:
        cpma.describe_optimiser(stub, run)
        rid = run.run_id

    m = store.load(rid)
    assert m.get("param.data.n_participants") == "2"
    # len() on a DataFrameGroupBy counts groups, not rows.
    assert m.get("param.data.n_records") == "6"
    assert m.get("param.fit.number_of_starts") == "2"


def test_cpm_describe_data_handles_every_shape(store):
    """DataFrame, DataFrameGroupBy and list must all work without raising."""
    pd = pytest.importorskip("pandas")

    from daftar.adapters import cpm as cpma

    full = pd.DataFrame({"ppt": [1, 1, 2, 2], "x": [1.0, 2.0, 3.0, 4.0]})
    for obj, expected in (
        (full, "2"),
        (full.groupby("ppt"), "2"),
        ([{"a": 1}, {"a": 2}, {"a": 3}], "3"),
        (full[["x"]], None),          # no identifier column: report nothing
    ):
        with daftar.track("shape", store=store) as run:
            cpma.describe_data(obj, run)
            rid = run.run_id
        assert store.load(rid).get("param.data.n_participants") == expected


def test_replay_emits_a_runnable_pip_command(store):
    """A package from git cannot be pinned by version -- the version is not the code.

    Regression: `env.<pkg>.source` fields introduced in 0.1.3 were rendered as if
    they were packages, producing lines like
    `pip install jaxley.source==git+https://...`, which is not installable.
    """
    from daftar.manifest import Manifest

    m = Manifest(run_id="r-mix")
    m.set("code.entrypoint", "sim.py::main")
    m.set("seed.value", "1")
    m.set("env.numpy", "2.4.6")
    m.set("env.jaxley", "0.13.0")
    m.set("env.jaxley.source", "git+https://example.com/j.git#abc123def456")
    m.set("env.local_pkg", "1.0.0")
    m.set("env.local_pkg.source", "editable:file:///home/me/work")
    store.save(m)

    plan = daftar.plan_replay(store.load("r-mix"), check_current=False)
    text = plan.render()

    # The .source pseudo-entries must never appear as requirements.
    assert ".source==" not in text
    assert "jaxley.source" not in text
    # Versioned packages pin normally; VCS packages carry their origin.
    assert "numpy==2.4.6" in text
    assert '"jaxley @ git+https://example.com/j.git@abc123def456"' in text
    # Local installs are unfetchable and must be called out, not faked.
    assert "not fetchable by pip" in text
    assert "local_pkg==1.0.0" not in text
    assert any("local path" in w for w in plan.warnings)

    assert plan.sources["jaxley"].startswith("git+")
    assert "jaxley.source" not in plan.env
