"""Notebook provenance, tested against a real IPython shell.

An ``InteractiveShell`` instance behaves like a kernel for our purposes: it
maintains ``execution_count`` and ``history_manager.input_hist_raw`` exactly as
Jupyter and Colab do. Mocking those would test the mock.
"""

from __future__ import annotations

import zipfile

import pytest

import daftar
from daftar import notebook
from daftar.store import RunStore

IPython = pytest.importorskip("IPython")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAFTAR_DIR", str(tmp_path / ".daftar"))
    s = RunStore()
    s.init()
    return s


@pytest.fixture
def shell():
    """A live IPython shell, reset between tests."""
    from IPython.core.interactiveshell import InteractiveShell

    sh = InteractiveShell.instance()
    sh.history_manager.reset()
    sh.execution_count = 1
    yield sh
    InteractiveShell.clear_instance()


def _run(shell, code):
    shell.run_cell(code, store_history=True)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_no_notebook_fields_outside_ipython(store):
    """A plain script must not grow notebook fields."""
    assert notebook.notebook_context() == {}
    with daftar.track("script", store=store) as run:
        rid = run.run_id
    m = store.load(rid)
    assert m.get("code.notebook") is None
    assert "::" in m.get("code.entrypoint") or m.get("code.entrypoint")


def test_detects_a_live_shell(shell):
    assert notebook.in_notebook()
    ctx = notebook.notebook_context()
    assert ctx["notebook"] is True
    assert ctx["kernel"]


# --------------------------------------------------------------------------
# the two hashes that matter
# --------------------------------------------------------------------------

def test_cell_source_is_hashed_not_the_file(shell, store):
    """The record must survive the cell being edited afterwards."""
    _run(shell, "a = 1")
    _run(shell, "b = a + 1")

    with daftar.track("nb", store=store, _cell_source="result = b * 3") as run:
        rid = run.run_id

    m = store.load(rid)
    assert m.get("code.notebook") == "true"
    assert len(m.get("code.cell_sha256")) == 16
    assert m.get("code.cell_source") == "result = b * 3"

    # Editing the cell afterwards cannot change what was recorded.
    _run(shell, "result = b * 999")
    assert store.load(rid).get("code.cell_source") == "result = b * 3"


def test_identical_cells_with_different_history_are_different_runs(shell, store):
    """A notebook result depends on every cell executed before it.

    This is the property no other tool records, and the reason two people at the
    same git commit can hold different numbers.
    """
    _run(shell, "scale = 1.0")
    with daftar.track("nb", store=store, _cell_source="y = scale * 10") as run:
        first = run.run_id

    _run(shell, "scale = 2.0")          # a cell run in between
    _run(shell, "extra = 'something'")
    with daftar.track("nb", store=store, _cell_source="y = scale * 10") as run:
        second = run.run_id

    a, b = store.load(first), store.load(second)

    # Same cell.
    assert a.get("code.cell_sha256") == b.get("code.cell_sha256")
    # Different session state, and daftar says so.
    assert a.get("code.session_history_sha256") != b.get("code.session_history_sha256")
    assert int(a.get("code.session_n_cells")) < int(b.get("code.session_n_cells"))

    causes = [c.key for c in daftar.diff_manifests(a, b).causes]
    assert "code.session_history_sha256" in causes


def test_same_history_and_same_cell_compare_equal(shell, store):
    """The converse: no spurious differences from being in a notebook."""
    _run(shell, "k = 5")
    ids = []
    for _ in range(2):
        with daftar.track("nb", seed=1, store=store,
                          _cell_source="z = k + 1") as run:
            ids.append(run.run_id)

    d = daftar.diff_manifests(store.load(ids[0]), store.load(ids[1]))
    assert not d.causes, [c.key for c in d.causes]


def test_execution_count_is_metadata_not_a_cause(shell, store):
    """It increments on every run, so under code.* it would explain nothing.

    Kept -- it locates the cell in the session -- but in meta.*, which the diff
    treats as neutral. Otherwise every notebook comparison would list it as a
    candidate cause of a changed result.
    """
    for i in range(3):
        _run(shell, f"v{i} = {i}")

    ids = []
    for _ in range(2):
        _run(shell, "spacer = 1")
        with daftar.track("nb", seed=1, store=store,
                          _cell_source="same = 1") as run:
            ids.append(run.run_id)

    a, b = store.load(ids[0]), store.load(ids[1])
    assert int(a.get("meta.cell_execution_count")) >= 3
    assert a.get("code.cell_execution_count") is None
    assert a.get("meta.cell_execution_count") != b.get("meta.cell_execution_count")

    # The counts differ, but nothing is reported as a cause.
    causes = [c.key for c in daftar.diff_manifests(a, b).causes]
    assert "meta.cell_execution_count" not in causes


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def test_entrypoint_identifies_the_cell_and_is_stable(shell, store):
    """`<ipython-input-5-a1b2c3>` identifies nothing, and In[N] changes on every
    re-run -- which would make the entrypoint a spurious cause in every diff.
    The cell hash is stable across re-runs and distinct between cells."""
    _run(shell, "q = 1")

    ids = []
    for _ in range(2):
        _run(shell, "spacer = 2")
        with daftar.track("nb", seed=1, store=store,
                          _cell_source="work = q + 1") as run:
            ids.append(run.run_id)

    entries = [store.load(i).get("code.entrypoint") for i in ids]
    for e in entries:
        assert e.startswith("notebook::cell[")
        assert "ipython-input" not in e and "/tmp" not in e
    assert entries[0] == entries[1], "same cell must give a stable entrypoint"

    # A different cell must give a different entrypoint.
    with daftar.track("nb", seed=1, store=store,
                      _cell_source="other = q * 99") as run:
        other_id = run.run_id           # manifest is written on exit, not here
    other = store.load(other_id).get("code.entrypoint")
    assert other != entries[0]


# --------------------------------------------------------------------------
# downstream: replay and export
# --------------------------------------------------------------------------

def test_replay_plan_warns_about_session_dependence(shell, store):
    _run(shell, "setup = True")
    with daftar.track("nb", store=store, _cell_source="out = 1") as run:
        rid = run.run_id

    plan = daftar.plan_replay(store.load(rid), check_current=False)
    text = " ".join(plan.warnings)
    assert "notebook cell" in text
    assert "execution order" in text


def test_export_bundle_contains_the_cell_that_ran(shell, tmp_path, store):
    source = "import numpy as np\nout = np.arange(10).mean()"
    _run(shell, "prep = 1")
    with daftar.track("nb", store=store, _cell_source=source) as run:
        run.log_result("out", 4.5)
        rid = run.run_id

    bundle = daftar.export_bundle(store.load(rid), tmp_path / "nb.zip", store=store)
    with zipfile.ZipFile(bundle) as zf:
        assert "cell.py" in zf.namelist()
        assert zf.read("cell.py").decode() == source


# --------------------------------------------------------------------------
# magic line parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line,label,kwargs", [
    ("", "", {}),
    ("mylabel", "mylabel", {}),
    ("mylabel seed=42", "mylabel", {"seed": "42"}),
    ("seed=42", "", {"seed": "42"}),
    ("fit seed=7 dt=0.025", "fit", {"seed": "7", "dt": "0.025"}),
])
def test_magic_line_parsing(line, label, kwargs):
    assert notebook._parse_magic_line(line) == (label, kwargs)


def test_magics_class_builds():
    """The magic is constructed lazily; make sure it actually constructs."""
    cls = notebook._make_magics()
    assert cls.__name__ == "DaftarMagics"
    assert hasattr(cls, "daftar")
