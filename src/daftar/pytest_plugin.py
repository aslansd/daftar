"""pytest plugin — track computational results from inside a test suite.

Two things this makes possible that were awkward before.

**Tests as the place you record runs.** A test is already a named, repeatable
entry point with fixed inputs. Wrapping one in ``daftar.track()`` by hand works
but means repeating the label and the store wiring in every test; the
``daftar_run`` fixture does it once, labelling each run with the test's node id.

**Results as a regression check.** ``daftar diff`` already exits 1 when a run
fails to reproduce, which makes it usable in CI at the shell level. This plugin
moves that inside pytest: with ``--daftar-compare``, a test fails if its
``result.*`` fields differ from the last recorded run of the same test. Not
because the assertions broke — because the *numbers moved*.

That is a different kind of test from the ones people usually write. A unit test
says "this function returns 4". A daftar regression check says "this simulation
returns whatever it returned last time, and if that changed, something in the
code or the environment changed and you should know which". It catches the class
of failure this whole package exists for: a dependency upgrade that quietly moves
a result while every assertion still passes.

Usage::

    def test_my_simulation(daftar_run):
        result = simulate(dt=0.025)
        daftar_run.log_param("dt", 0.025)
        daftar_run.log_result("mean", float(result.mean()))

    $ pytest                      # records a run per test
    $ pytest --daftar-compare     # also fails if results moved
    $ pytest --daftar-update      # accept the new numbers as the baseline
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from .diff import diff_manifests
from .run import Run, _TrackContext
from .store import RunStore

#: Marker set on the manifest so plugin-created runs are identifiable later.
PYTEST_TAG = "pytest"


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("daftar", "computational result provenance")
    group.addoption(
        "--daftar-dir",
        action="store",
        default=None,
        metavar="PATH",
        help="run store directory (default: nearest .daftar)",
    )
    group.addoption(
        "--daftar-compare",
        action="store_true",
        default=False,
        help="fail a test if its result.* fields differ from the last "
             "recorded run of the same test",
    )
    group.addoption(
        "--daftar-update",
        action="store_true",
        default=False,
        help="with --daftar-compare, accept the new results as the baseline "
             "instead of failing",
    )
    group.addoption(
        "--daftar-seed",
        action="store",
        type=int,
        default=None,
        metavar="N",
        help="seed applied to every tracked test run",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "daftar(label=None, seed=None): configure the daftar_run fixture "
        "for this test",
    )


class _Comparison:
    """Result of comparing a run against the previous run of the same test."""

    def __init__(self, previous: Any = None, changes: list | None = None):
        self.previous = previous
        self.changes = changes or []

    @property
    def regressed(self) -> bool:
        return bool(self.changes)


def _previous_run(store: RunStore, label: str, exclude_id: str):
    """The most recent *accepted* run with this label, other than this one.

    Runs that themselves failed the comparison are skipped. Without that, a
    regression would become the new baseline the moment it was reported, the
    next run would compare against the changed value and pass, and the check
    would fire exactly once before going quiet -- which is worse than not
    having it, because you would believe it was watching.

    The baseline only moves when you accept it with ``--daftar-update``.
    """
    for manifest in store.list(label=label):
        if manifest.run_id == exclude_id:
            continue
        if manifest.get("meta.status") != "completed":
            continue
        if manifest.get("meta.daftar_regression") == "true":
            continue
        return manifest
    return None


def _compare(store: RunStore, manifest: Any, label: str) -> _Comparison:
    previous = _previous_run(store, label, manifest.run_id)
    if previous is None:
        return _Comparison()
    d = diff_manifests(previous, manifest)
    # Only results matter here. A changed environment is information, not a
    # failure -- and reporting it as one would make the check unusable on any
    # machine that ever upgrades anything.
    return _Comparison(previous=previous, changes=d.effects)


def _format_regression(label: str, comparison: _Comparison) -> str:
    lines = [
        f"daftar: results changed for {label}",
        f"  baseline run: {comparison.previous.run_id} "
        f"({comparison.previous.started_at})",
        "",
    ]
    width = max(len(c.key) for c in comparison.changes)
    for change in comparison.changes:
        lines.append(f"  {change.key.ljust(width)}  {change.a}  ->  {change.b}")
    lines += [
        "",
        "  The assertions in this test passed; the numbers moved anyway.",
        "  Run `daftar diff` on the two ids above to see what else changed --",
        "  a dependency upgrade will show up there as a candidate cause.",
        "",
        "  If the new values are correct, re-run with --daftar-update.",
    ]
    return "\n".join(lines)


@pytest.fixture
def daftar_run(request: Any):
    """A tracked :class:`daftar.Run` labelled with the test's node id.

    ::

        def test_simulation(daftar_run):
            daftar_run.log_param("dt", 0.025)
            daftar_run.log_result("mean_rate", 4.81)

    Override the label or seed per test with the ``daftar`` marker::

        @pytest.mark.daftar(label="my-sim", seed=42)
        def test_simulation(daftar_run):
            ...
    """
    config = request.config
    marker = request.node.get_closest_marker("daftar")
    options = dict(marker.kwargs) if marker else {}

    label = options.get("label") or request.node.nodeid
    seed = options.get("seed", config.getoption("--daftar-seed"))

    store_dir = config.getoption("--daftar-dir")
    store = RunStore(store_dir) if store_dir else RunStore()

    context = _TrackContext(Run(
        label=label,
        seed=seed,
        store=store,
        tags=[PYTEST_TAG],
        entrypoint=request.node.nodeid,
    ))

    run = context.__enter__()
    run.manifest.set("meta.pytest_nodeid", request.node.nodeid)

    # The run is closed and compared by pytest_runtest_call below, not here.
    # Doing it in fixture teardown would make a regression a teardown *error*
    # ("1 passed, 1 error"), which reads as a broken fixture rather than a
    # result that moved. Closing during the call phase makes it a plain failure.
    request.node._daftar = {"context": context, "run": run,
                            "store": store, "label": label}
    yield run


def _finalise(item: Any, failed: bool) -> None:
    """Close the run and, when asked, compare it against the previous one."""
    state = getattr(item, "_daftar", None)
    if state is None:
        return
    item._daftar = None

    context, run, store, label = (state["context"], state["run"],
                                  state["store"], state["label"])
    if failed:
        context.__exit__(AssertionError, AssertionError("test failed"), None)
        return
    context.__exit__(None, None, None)

    config = item.config
    if not config.getoption("--daftar-compare"):
        return

    manifest = store.load(run.run_id)
    comparison = _compare(store, manifest, label)
    if not comparison.regressed:
        return

    if config.getoption("--daftar-update"):
        # Accepted: this run becomes the baseline for the next comparison.
        manifest.set("meta.daftar_baseline_accepted", True)
        store.save(manifest)
        return

    # Rejected: mark it so it does not silently become the next baseline.
    manifest.set("meta.daftar_regression", True)
    store.save(manifest)
    pytest.fail(_format_regression(label, comparison), pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: Any):
    """Close the tracked run inside the call phase.

    A regression then surfaces as a test failure rather than a teardown error,
    which is what it actually is: the code ran fine and produced different
    numbers.

    New-style ``wrapper=True`` rather than ``hookwrapper=True`` because this
    hook deliberately raises ``Failed`` after the test body; pluggy warns about
    exceptions from old-style wrapper teardown.
    """
    try:
        result = yield
    except BaseException:
        _finalise(item, failed=True)
        raise
    _finalise(item, failed=False)   # may raise Failed for a regression
    return result


def pytest_report_header(config: Any) -> str | None:
    """Say where runs are going, so a green suite is not silently recording."""
    store_dir = config.getoption("--daftar-dir")
    store = RunStore(store_dir) if store_dir else RunStore()
    mode = []
    if config.getoption("--daftar-compare"):
        mode.append("compare")
    if config.getoption("--daftar-update"):
        mode.append("update")
    suffix = f" [{', '.join(mode)}]" if mode else ""
    return f"daftar: run store {store.dir}{suffix}"


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int,
                            config: Any) -> None:
    """Report how many runs this session recorded, and where."""
    store_dir = config.getoption("--daftar-dir")
    store = RunStore(store_dir) if store_dir else RunStore()
    if not store.runs_dir.exists():
        return
    try:
        recorded = [m for m in store.list(limit=None)
                    if PYTEST_TAG in (m.get("meta.tags") or "")]
    except Exception:
        return
    if not recorded:
        return
    terminalreporter.write_sep("-", "daftar")
    terminalreporter.write_line(
        f"{len(recorded)} tracked run(s) in {store.dir}"
    )
    terminalreporter.write_line(
        "compare against the previous run with: pytest --daftar-compare"
        if not config.getoption("--daftar-compare")
        else "results compared against the previous run of each test"
    )
