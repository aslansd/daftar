# Testing daftar

## The short version

```bash
pip install -e ".[dev]"
pytest -q
```

What you should see depends on which frameworks are installed:

| Environment | Result |
|---|---|
| daftar only | **60 passed, 35 skipped** — the core and notebook suites, plus the adapter status report which always runs |
| Environment A (brian2, jaxley, cpm, mne, sbi, nilearn, gdsfactory, netpyne) | **93 passed, 2 skipped** — only MeltingPot missing |
| Environment B (meltingpot) | **62 passed, 33 skipped** |

**The core and notebook suites must always pass — 59 tests, no exceptions.**
Every skip should be a live adapter test whose framework is absent. If anything
in `test_core.py` or `test_notebook.py` skips, something is wrong.

---

## The three suites

| File | Tests | Needs | What it covers |
|---|---|---|---|
| `tests/test_core.py` | 44 | nothing | manifests, capture, diff verdicts, sweeps, replay, export, store |
| `tests/test_notebook.py` | 15 | `ipython` | cell hashing, session history, the `%%daftar` magic |
| `tests/test_adapters_live.py` | 36 | the frameworks | real workloads through each adapter |

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
