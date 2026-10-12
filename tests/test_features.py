"""Tests for the pytest plugin, notebook staleness detection, and run browser."""

from __future__ import annotations

import json
import re

import pytest

import daftar
from daftar.store import RunStore

pytest_plugins = ["pytester"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAFTAR_DIR", str(tmp_path / ".daftar"))
    s = RunStore()
    s.init()
    return s


# ==========================================================================
# pytest plugin
# ==========================================================================

_SIM = '''
import os
FACTOR = float(os.environ.get("FACTOR", "1.0"))

def test_sim(daftar_run):
    out = 100.0 * FACTOR
    daftar_run.log_param("n", 100)
    daftar_run.log_result("total", round(out, 6))
    assert out > 0
'''


def test_plugin_records_a_run_per_test(pytester):
    pytester.makepyfile(test_sim=_SIM)
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)

    store = RunStore(pytester.path / ".daftar")
    runs = store.list()
    assert len(runs) == 1
    m = runs[0]
    assert m.get("meta.status") == "completed"
    assert m.get("result.total") == "100.0"
    assert m.get("param.n") == "100"
    # Labelled with the node id, so runs are attributable to a test.
    assert "test_sim" in m.label
    assert m.get("meta.pytest_nodeid")


def test_plugin_fails_when_results_move(pytester, monkeypatch):
    """The assertions still pass; the numbers moved. That is the whole point."""
    pytester.makepyfile(test_sim=_SIM)
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(passed=1)

    monkeypatch.setenv("FACTOR", "1.03")
    result = pytester.runpytest("-q", "--daftar-compare")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*results changed*", "*result.total*100.0*103.0*"])


def test_plugin_regression_is_a_failure_not_a_teardown_error(pytester, monkeypatch):
    """A moved result must read as a failing test, not a broken fixture.

    Finalising the run in fixture teardown produced "1 passed, 1 error", which
    looks like the harness is broken rather than the result having changed.
    """
    pytester.makepyfile(test_sim=_SIM)
    pytester.runpytest("-q", "--daftar-compare")
    monkeypatch.setenv("FACTOR", "2.0")
    result = pytester.runpytest("-q", "--daftar-compare")
    result.assert_outcomes(failed=1)          # not errors=1
    assert "error" not in result.outlines[-1].lower()


def test_plugin_rejected_run_does_not_become_the_baseline(pytester, monkeypatch):
    """Otherwise the check fires once and then goes quiet -- worse than absent."""
    pytester.makepyfile(test_sim=_SIM)
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(passed=1)

    monkeypatch.setenv("FACTOR", "1.03")
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(failed=1)
    # Same wrong value again: must still fail.
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(failed=1)

    # Explicit acceptance moves the baseline.
    pytester.runpytest("-q", "--daftar-compare", "--daftar-update") \
        .assert_outcomes(passed=1)
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(passed=1)


def test_plugin_environment_change_alone_is_not_a_regression(pytester):
    """Only result.* fields fail the check.

    A changed environment is information, not a failure -- flagging it would
    make the check unusable on any machine that ever upgrades anything.
    """
    pytester.makepyfile(test_sim=_SIM)
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(passed=1)
    pytester.runpytest("-q", "--daftar-compare").assert_outcomes(passed=1)

    store = RunStore(pytester.path / ".daftar")
    a, b = store.list()[0], store.list()[1]
    assert a.get("result.total") == b.get("result.total")


def test_plugin_failing_test_records_the_run_as_failed(pytester):
    pytester.makepyfile(test_bad='''
def test_bad(daftar_run):
    daftar_run.log_param("x", 1)
    assert False, "boom"
''')
    pytester.runpytest("-q").assert_outcomes(failed=1)
    runs = RunStore(pytester.path / ".daftar").list()
    assert len(runs) == 1
    assert runs[0].get("meta.status") == "failed"
    assert runs[0].get("param.x") == "1"


def test_plugin_marker_overrides_label_and_seed(pytester):
    pytester.makepyfile(test_marked='''
import pytest

@pytest.mark.daftar(label="custom-label", seed=123)
def test_marked(daftar_run):
    daftar_run.log_result("v", 1)
''')
    pytester.runpytest("-q").assert_outcomes(passed=1)
    m = RunStore(pytester.path / ".daftar").list()[0]
    assert m.label == "custom-label"
    assert m.get("seed.value") == "123"
    assert m.get("seed.was_explicit") == "true"


# ==========================================================================
# notebook: cells edited and re-run
# ==========================================================================

def test_session_redefinition_detects_an_edited_cell():
    from daftar.notebook import session_redefinitions

    history = [
        "import numpy as np",
        "def simulate(dt):\n    return dt * 2",
        "scale = 1.0",
        "out = simulate(0.5)",
        "def simulate(dt):\n    return dt * 3",     # edited and re-run
    ]
    result = session_redefinitions(history)
    assert result["names"] == ["simulate"]
    assert result["counts"]["simulate"] == 2
    assert result["n_redefined"] == 1


def test_session_redefinition_ignores_identical_reruns():
    """Re-running a cell unchanged is normal and must not be flagged."""
    from daftar.notebook import session_redefinitions

    history = ["scale = 1.0", "x = 2", "scale = 1.0"]
    assert session_redefinitions(history)["n_redefined"] == 0


def test_session_redefinition_survives_syntax_errors():
    """A cell that failed to parse must not take the whole probe down."""
    from daftar.notebook import session_redefinitions

    history = ["a = 1", "this is not python !!!", "a = 2"]
    assert session_redefinitions(history)["names"] == ["a"]


def test_assigned_names_covers_the_common_bindings():
    from daftar.notebook import _assigned_names

    assert _assigned_names("x = 1") == frozenset({"x"})
    assert _assigned_names("def f(): pass") == frozenset({"f"})
    assert _assigned_names("class C: pass") == frozenset({"C"})
    assert _assigned_names("import numpy as np") == frozenset({"np"})
    assert _assigned_names("from x import y") == frozenset({"y"})
    assert _assigned_names("x: int = 1") == frozenset({"x"})
    # Not a binding, so not an identity.
    assert _assigned_names("print(x)") == frozenset()


def test_notebook_context_flags_stale_risk():
    """A run made after an edited-and-rerun cell carries the warning."""
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    from daftar import notebook

    shell = InteractiveShell.instance()
    shell.history_manager.reset()
    try:
        shell.run_cell("def simulate(dt):\n    return dt * 2", store_history=True)
        shell.run_cell("out = simulate(0.5)", store_history=True)
        shell.run_cell("def simulate(dt):\n    return dt * 3", store_history=True)

        ctx = notebook.notebook_context(cell_source="final = simulate(1.0)")
        assert ctx["session_stale_risk"] is True
        assert "simulate" in ctx["session_redefined"]
        assert ctx["session_n_redefined"] == 1
    finally:
        InteractiveShell.clear_instance()


# ==========================================================================
# run browser
# ==========================================================================

def _seed_runs(store, n=3):
    for i in range(n):
        with daftar.track("browse-demo", params={"i": i}, seed=1,
                          store=store) as run:
            run.log_result("value", i * 2)
    return store


def test_browser_html_is_self_contained(store, tmp_path):
    """It must keep working when emailed, attached to a paper, or opened in 2031."""
    from daftar.browse import write_html

    _seed_runs(store)
    out = write_html(store, tmp_path / "runs.html")
    text = out.read_text(encoding="utf-8")

    # No external anything.
    assert "<script src" not in text
    assert "<link " not in text
    assert 'src="http' not in text and 'href="http' not in text
    assert text.lstrip().startswith("<!DOCTYPE html>")


def test_browser_embeds_every_manifest_field(store, tmp_path):
    from daftar.browse import build_html

    _seed_runs(store, n=2)
    html = build_html(store)

    payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html,
                                   re.S).group(1))
    assert len(payload["runs"]) == 2
    first = payload["runs"][0]
    assert "result.value" in first["fields"]
    assert "param.i" in first["fields"]
    # The cause/effect split travels with the data so the page can diff offline.
    assert "code" in payload["causes"] and "result" in payload["effects"]


def test_browser_respects_label_and_limit(store, tmp_path):
    from daftar.browse import build_html

    _seed_runs(store, n=3)
    with daftar.track("other", store=store) as run:
        run.log_result("z", 1)

    payload_all = json.loads(re.search(r"const DATA = (\{.*?\});\n",
                                       build_html(store), re.S).group(1))
    assert len(payload_all["runs"]) == 4

    payload_lbl = json.loads(re.search(r"const DATA = (\{.*?\});\n",
                                       build_html(store, label="other"),
                                       re.S).group(1))
    assert len(payload_lbl["runs"]) == 1

    payload_lim = json.loads(re.search(r"const DATA = (\{.*?\});\n",
                                       build_html(store, limit=2),
                                       re.S).group(1))
    assert len(payload_lim["runs"]) == 2


def test_browse_command_writes_a_file(store, tmp_path, capsys):
    from daftar.cli import main

    _seed_runs(store)
    out = tmp_path / "browser.html"
    assert main(["browse", "-o", str(out)]) == 0
    assert out.exists()
    assert "Self-contained" in capsys.readouterr().out


def test_browse_command_handles_an_empty_store(tmp_path, monkeypatch, capsys):
    from daftar.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAFTAR_DIR", str(tmp_path / ".daftar"))
    assert main(["browse", "-o", str(tmp_path / "x.html")]) == 0
    assert "No runs recorded" in capsys.readouterr().out
