# Testing daftar

## The short version

```bash
pip install -e ".[dev]"
pytest -q
```

What you should see depends on which frameworks are installed:

| Environment | Result |
|---|---|
| daftar only | **85 passed, 35 skipped** — the core and notebook suites, plus the adapter status report which always runs |
| Environment A (brian2, jaxley, cpm, mne, sbi, nilearn, gdsfactory, netpyne) | **118 passed, 2 skipped** — only MeltingPot missing |
| Environment B (meltingpot) | **87 passed, 33 skipped** |

**The core, notebook and feature suites must always pass — 84 tests, no exceptions.**
The Concordia adapter is tested here rather than in the live suite: its wrapper
is duck-typed, so its divergence logic runs against a fake model and needs no
provider and no Concordia install.
Every skip should be a live adapter test whose framework is absent. If anything
in `test_core.py` or `test_notebook.py` skips, something is wrong.

---

## The three suites

| File | Tests | Needs | What it covers |
|---|---|---|---|
| `tests/test_core.py` | 52 | nothing | manifests, capture, diff verdicts, sweeps, replay, export, store, **and the Concordia adapter** |
| `tests/test_notebook.py` | 15 | `ipython` | cell hashing, session history, the `%%daftar` magic |
| `tests/test_adapters_live.py` | 36 | the frameworks | real workloads through each adapter |
| `tests/test_features.py` | 17 | nothing | the pytest plugin, notebook staleness detection, the run browser |

The core and notebook suites must always pass. They have no optional
dependencies beyond IPython and they are fast.

---

## Running the live adapter tests

These are separated because they need heavy optional dependencies and are slow.
They are also the **only** tests that can catch adapter rot.

```bash
# Environment A: brian2, jaxley, cpm, mne, sbi, nilearn, gdsfactory, netpyne
conda activate daftar312
pytest tests/test_adapters_live.py -v

# Environment B: meltingpot
conda activate daftar311
pytest tests/test_adapters_live.py -v
```

See [INSTALL.md](INSTALL.md) for why two environments are needed.

### Why they exist

Adapters fail **silently by design**. `adapters/base.py` wraps every probe in
`safe()`, so a renamed attribute in a target framework produces `<unavailable>`
in the manifest rather than crashing a four-hour simulation. That is the right
behaviour, and it means a broken adapter looks fine until someone reads a
manifest months later.

Only a live run against a real object notices. **Re-run this suite after every
upgrade of a target framework**, not only at release.

### Reading the skips

```bash
pytest tests/test_adapters_live.py -rs
```

`-rs` prints skip reasons, which `-q` and `-v` both hide. There are three kinds
and they mean different things:

| Skip reason | Meaning |
|---|---|
| `X not installed` | Ordinary. Install it or ignore it. |
| `X is INSTALLED BUT BROKEN -- <exception>` | The framework will not import. Fixable — see TROUBLESHOOTING.md. |
| `X/Y version clash: ... Not a daftar bug. Fix with: ...` | The framework imports but cannot run against its own dependency. The message names the fix. |

The rule the suite follows: **skip for a clash we recognise and can explain,
fail loudly for anything we do not.** An unrecognised error might be a daftar
bug, and a silent skip would hide it. A skip that says nothing is worse than a
red test.

### The status report

```bash
pytest tests/test_adapters_live.py -k at_least_report -s
```

Never fails. Prints the adapter matrix, the framework versions it would be
testing against, and a runtime usability line per framework — because "the
adapter tests passed" only means something alongside what they passed against.

---

## Testing the Concordia adapter

It is the one adapter tested in the core suite rather than the live suite:

```bash
pytest tests/ -k concordia -v          # 7 tests, no install required
```

The wrapper delegates to a language model by attribute lookup rather than
subclassing `LanguageModel`, so its logic runs against a fake model with no
`gdm-concordia` and no provider credentials. That is deliberate — an adapter
whose tests need an API key is an adapter nobody runs the tests for.

What those tests cover, and what to check if you change it:

| Test | Guards |
|---|---|
| `records_the_trajectory_not_a_replay_promise` | the manifest states plainly that an unseeded run is not repeatable, rather than implying it is |
| `identical_runs_have_identical_trajectories` | the chain hash is stable when nothing changed |
| `locates_a_response_divergence` | same prompt, different answer → reported as provider non-determinism |
| `locates_a_prompt_divergence` | different prompt → reported as upstream simulation divergence |
| `detects_a_length_divergence` | one run made more calls than the other |
| `wrapper_delegates_unknown_attributes` | it is not a `LanguageModel` subclass; anything unimplemented passes through |
| `transcript_is_written_to_a_file_not_the_manifest` | a planted confidential string never reaches the manifest |

The last one matters most. Prompts in agent simulations routinely contain the
entire scenario, so the manifest gets fixed-width hashes and the full text goes
to a file the export bundle carries. If you change what the adapter records,
that test is the one that stops sensitive content leaking into a record meant to
be committed to git.

### Checking divergence reporting by hand

```python
from daftar.adapters import concordia as ca
a, b = store.load(run_a), store.load(run_b)
print(ca.render_divergence(ca.first_divergence(a, b)))
```

Against two runs of the same scenario this should either report identical
trajectories or name the exact call at which they split. If it reports a
divergence at call 0, the runs were never comparable in the first place.

---

## Using daftar in your own test suite

The plugin is installed with daftar — no separate package:

```python
def test_my_simulation(daftar_run):
    daftar_run.log_param("dt", 0.025)
    daftar_run.log_result("mean", 4.81)
```

```bash
pytest                                  # record a run per test
pytest --daftar-compare                 # fail if result.* moved
pytest --daftar-update                  # accept new numbers as baseline
pytest --daftar-dir /path/to/store      # somewhere other than ./.daftar
pytest --daftar-seed 42                 # seed every tracked test run
```

Per-test overrides use the marker:

```python
@pytest.mark.daftar(label="my-sim", seed=42)
def test_simulation(daftar_run):
    ...
```

### The properties worth preserving

`tests/test_features.py` pins three behaviours that were each got wrong once:

1. **A regression is a test failure, not a teardown error.** Finalising the run
   in fixture teardown produced `1 passed, 1 error`, which reads as a broken
   fixture rather than a result that moved. The run is closed during the call
   phase instead.
2. **A rejected run does not become the next baseline.** Otherwise the check
   fires once, the changed value becomes the reference, and it goes quiet — worse
   than not having it, because you would believe it was watching.
3. **Only `result.*` fields fail the check.** A changed environment is
   information, not a failure; flagging it would make the check unusable on any
   machine that ever upgrades anything.

---

## The end-to-end demo

```bash
python examples/demo_end_to_end.py
```

No frameworks needed. Walks through a clean reproduction, a deliberate parameter
change, a genuine nondeterminism catch, a sweep, a replay plan, and an export.
If this runs on a fresh clone, the package works.

```bash
python examples/adapter_usage.py
```

Prints which adapters are available here and the reference usage for each.

---

## Checking the notebook path

The notebook tests drive a real `InteractiveShell` rather than mocks, so
`pytest` covers them. To exercise the magic interactively:

```bash
jupyter notebook examples/daftar_in_notebooks.ipynb
```

or open the same file in Colab. Section 3 is the one worth running: an identical
cell executed twice with a changed upstream variable, which daftar attributes to
`code.session_history_sha256` rather than calling nondeterministic.

---

## Before a release

```bash
pytest -q                                    # core + notebook, all green
pytest tests/test_adapters_live.py -v        # in each environment
python examples/demo_end_to_end.py           # runs clean
daftar doctor                                # exit 0
python -m build && python -m twine check dist/*
```

Then install the built wheel into a fresh virtualenv and import it — that
catches packaging mistakes the test suite cannot see:

```bash
python -m venv /tmp/check
/tmp/check/bin/pip install dist/daftar-*.whl
/tmp/check/bin/python -c "import daftar, sys; print(daftar.__version__)"
/tmp/check/bin/daftar doctor
```

See [PUBLISHING.md](PUBLISHING.md) for the upload steps.
