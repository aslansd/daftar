# daftar

**دفتر** — *ledger, register, record-book.* The same word in Persian, Turkish
(*defter*), Azerbaijani, Arabic, Urdu, and Hindi.

Record what produced each computational result — the environment, code version,
parameters, random seeds, and input files — so two runs can be compared and any
past run can be rebuilt.

Experiment trackers solved this for deep learning, where an experiment is
`model.fit()`. They do not help when the experiment is a Hodgkin–Huxley
simulation of 302 neurons, a hierarchical fit across 60 participants, an EEG
preprocessing pipeline, a photonic layout script, or a 200-episode multi-agent
sweep. Those runs have no epochs, no loss
curves, and no checkpoints. They have parameter grids, solver tolerances, random
seeds, and derived quantities.

**No dependencies. Runs offline. No account, no server, no network call.**

```bash
pip install daftar
```

Python 3.10+. Apache 2.0.

[PyPI](https://pypi.org/project/daftar/) · [INSTALL](INSTALL.md) ·
[TESTING](TESTING.md) · [PUBLISHING](PUBLISHING.md) ·
[TROUBLESHOOTING](TROUBLESHOOTING.md) · [ROADMAP](ROADMAP.md) ·
[CHANGELOG](CHANGELOG.md)

---

## The thirty-second version

```python
import daftar   # no alias: `df` would collide with the pandas convention

with daftar.track("celegans", params={"dt": 0.025, "solver": "bwd_euler"}, seed=42) as run:
    run.add_input("data/connectome.csv")
    v = simulate(dt=0.025)
    run.log_result("mean_rate_hz", float(v.mean()))
```

Then, next Thursday, when the number is different:

```
$ daftar diff r-4f21ab r-88c07e

--- r-4f21ab  celegans  2026-05-14T14:02:11+00:00
+++ r-88c07e  celegans  2026-05-16T09:47:03+00:00

candidate causes (1)
  env.jax               0.4.35  ->  0.4.41

observed effects (1)
  result.mean_rate_hz   4.812  ->  4.796

2 meaningful field(s) differ, 27 identical

Results differ, and so do things that could explain it.
The candidate causes below are where to look.
```

Same code, same seed, same inputs. A patch-level JAX upgrade moved the answer.
That is the question this tool exists to answer, and it answers it the same
afternoon you install it.

---

## Why this and not the eight tools that came before

Binder, ReproZip, Sciunit, Whole Tale, Gigantum, Renku, Code Ocean: in a
published survey, exactly **one of 38 researchers** had ever used a dedicated
platform to save and re-run a computational experiment.

Every one of those tools asked scientists to change how they work today in
exchange for a benefit arriving in three years. That trade never closes.

`daftar` inverts it. `diff` answers a question people ask weekly — *why is this
number different from Tuesday's?* — and provenance arrives as a free side effect
of a tool they already wanted. Reproducibility is the by-product, not the pitch.

Three consequences of taking that seriously:

- **One line and one indent.** No DSL, no workflow language, no migration. If
  wrapping an existing simulation costs more than that, nobody does it.
- **The manifest is plain JSON with sorted, dotted keys.** Readable and
  greppable without this package installed, and `git diff`-able. If daftar
  disappears, the record survives.
- **Failed runs are recorded too.** A crashed run that took four hours is
  exactly the one you will want to look at later.

---

## What gets captured

| Namespace | Contents |
|---|---|
| `code.*` | git commit, branch, dirty flag, **hash of uncommitted changes**, entrypoint, argv; in notebooks, the **cell hash and session history hash** |
| `param.*` | everything you chose, including defaults you never passed |
| `seed.*` | seeds **applied** to `random`, numpy, torch, **Brian2's device RNG**; the JAX root key |
| `input.*` | sha256 and size of every declared input file or directory |
| `env.*` | interpreter, OS, package versions, **and where each package was installed from** |
| `result.*` | scalar outcomes worth comparing |
| `output.*` | sha256 of produced files |
| `cost.*` | wall clock, host, cpu count |

Three of these are less obvious than they look.

**Seeds are applied, not merely recorded.** A tool that only writes down the
seed is close to useless: if the code seeded itself from the clock, recording
that fact tells you the run is irreproducible but does nothing to fix it.
`track(seed=...)` sets every RNG it can reach — including Brian2's device RNG,
which numpy seeding does not touch — then records what it set. If you pass no
seed, one is generated, applied, and recorded.

**A version string is not an identity.** jaxley 0.13.0 on PyPI was broken with
current JAX; jaxley `main` fixed it and still called itself 0.13.0. Recording
only `env.jaxley = 0.13.0` would call those two environments identical when one
works and one does not. daftar reads PEP 610 `direct_url.json` and adds
`env.<pkg>.source` for anything not installed from an index. `daftar replay`
then emits a `pip install` line that actually works, pinning index packages by
version and VCS packages by URL and commit.

**Dirty working trees get their diff hashed.** "Dirty" alone tells you there
were uncommitted edits but not whether they were the *same* edits. Hashing the
diff means two dirty runs can still be proven identical, which is the common
case during a debugging session.

---

## Commands

```bash
daftar list                    # what has been run
daftar show r-4f21ab           # one run's full manifest
daftar diff r-4f21ab r-88c07e  # what changed, and whether it mattered
daftar vary -l my-sweep        # which fields differ across many runs
daftar replay r-4f21ab         # what it would take to reproduce this
daftar export r-4f21ab -o run.zip
daftar doctor                  # which adapters work here, and why not
```

`diff` exits 0 if the second run reproduces the first and 1 otherwise, so it
works in CI as a regression check on your own results. `doctor` exits 1 if any
adapter's framework is installed but broken.

### Verdicts

`diff` does not just list changed fields. It separates fields that could have
**caused** a difference (`code`, `param`, `seed`, `input`, `env`) from fields
that merely **record** one (`result`, `output`), and reports what the
combination implies:

| Verdict | Meaning |
|---|---|
| `identical` | Nothing meaningful moved. |
| `explained` | Results differ and so do plausible causes. Here they are. |
| `no_effect` | Environment changed, results didn't. Evidence of robustness. |
| **`nondeterministic`** | **Results differ and nothing that could have caused it does.** |
| `incomparable` | The runs recorded different result fields. |

That fourth verdict is the valuable one. An unseeded RNG buried three libraries
deep can survive for years because nobody ever compares two runs precisely
enough to notice. daftar reports it as a finding rather than a glitch.

---

## Sweeps

```python
result = daftar.sweep(
    simulate, label="tau-sweep", seed=42,
    dt=[0.025, 0.01], tau=[5.0, 10.0, 20.0],
)
print(result.table())
```

Each grid point is a separate run with its own manifest — not one run with a
nested table. That means a sweep point and a run you did by hand last Tuesday
are the same kind of object, and `diff` works across them. A sweep that fails at
point 3 of 40 keeps the first two results.

---

## Notebooks and Colab

Notebooks are the hardest case: the git commit means little when cells ran in an
unrecorded order, edited cells overwrite the code that made your figure, and on
Colab there is no repository at all.

`track()` detects IPython automatically — existing notebook code needs no
changes — and adds two fields:

| Field | What it pins down |
|---|---|
| `code.cell_sha256` | The cell source, hashed **before** execution, so it survives the cell being edited afterwards |
| `code.session_history_sha256` | Every cell executed before this one. A notebook result depends on the whole session, and nothing else records that |

That second field is the one that matters. Run the same cell twice with a
different upstream variable and daftar reports:

```
candidate causes (2)
  code.session_history_sha256  519aae2e385c2edd  ->  477357699f00be28
  code.session_n_cells         3  ->  6

observed effects (2)
  result.mean                  -0.00090  ->  -0.00272
  result.std                   1.00012   ->  3.00038
```

Without it, that pair would diff as `nondeterministic` — wrong, and it would
send you looking for a seeding bug that does not exist.

There is also a cell magic, which additionally stores the cell body verbatim so
the exported bundle contains the code that actually ran:

```python
%load_ext daftar

%%daftar montecarlo seed=42
vals = simulate(scale=scale)
run.log_result("mean", float(vals.mean()))
```

See `examples/daftar_in_notebooks.ipynb`, which runs on Colab.

---

## Replay

`daftar replay` prints a plan, and deliberately does **not** execute anything.
Re-running arbitrary recorded code would mean this package executes whatever a
manifest tells it to, and it still could not restore your CUDA driver. What it
does honestly is state the target state, check the current state against it, and
list every discrepancy:

```
Replay plan for r-4f21ab

  entrypoint   sim/celegans_hh.py::run_network
  commit       9c1d0ae
  seed         42

  BLOCKERS -- this run cannot be reproduced as recorded:
    - input file changed since the run: data/connectome.csv
      (was a7f39b21, now 3e0c77af)

  To reproduce:
    git checkout 9c1d0ae
    pip install numpy==2.4.6 "jaxley @ git+https://github.com/jaxleyverse/jaxley.git@2638cca"
```

Packages installed from a local path or in editable mode are listed as comments
rather than requirements, and raise a warning: nobody else can fetch
`file:///Users/you/Downloads/thing`, and a version pin would not reproduce it.

---

## Export

`daftar export` writes a zip containing `README.md`, `manifest.json`,
`fields.tsv`, and the referenced input and output files — plus `cell.py` for
notebook runs. The README is generated in plain English at the archive root, so
a successor learns what they are looking at without installing anything. A
bundle that needs our tool to be understood defeats its own purpose.

---

## Framework adapters

The core tracks any Python function. An adapter earns its existence only by
knowing something a generic tracker cannot infer.

| Adapter | Records what you'd otherwise lose |
|---|---|
| `jaxley` | morphology (compartments, branches, channels, synapses), `jx.integrate` defaults you never passed, `jax_enable_x64`, backend |
| `cpm` | parameter **bounds and priors** (resolved to the scipy distribution and its arguments), estimator and its scipy settings, restart counts and the initial guesses themselves, per-participant convergence, cohort hash |
| `brian2` | **the integration method Brian2 actually chose** — its default is a candidate list and the winner is stored nowhere — plus resolved `codegen.target`, network schedule, equation hashes, realised synapse counts |
| `gdsfactory` | **the content hash of the written GDS** — the mask a foundry receives, which carries no record of what made it — plus the active PDK and its layer map, the library stack that moves polygons between versions, and the design settings |
| `netpyne` | **whether the compiled NEURON mechanism library is stale** — a `.mod` edit without `nrnivmodl` silently runs the old binary — plus NetPyNE's four RNG seeds, `hParams`, the realised connection count, and per-section model hashes |
| `nilearn` | **which confounds were regressed out** (a runtime argument nilearn forgets), the mask that actually resolved, the atlas region count, GLM design columns, and `cov_estimator` resolving to Ledoit-Wolf |
| `sbi` | **the training hyperparameters sbi discards** and whether training converged or hit the epoch limit, the resolved density-estimator architecture, and the proposal each round drew from |
| `mne` | **which ICA components were excluded** and whether ICA converged, filter *design* rather than just the band, bad channels, epoch drop counts and reasons — with subject data hashed, never stored |
| `meltingpot` | resolved substrate ConfigDict hash, roles, episode-length cap, pinned bot checkpoints, per-player returns and Gini |

```python
from daftar.adapters import brian2 as b2a

with daftar.track("balanced-net", seed=42) as run:
    b2a.run_network(net, 1*second, run)
```

The MNE and nilearn adapters deserve a note on privacy: it runs against human neuroimaging
recordings, so they record structure and never content. Subject metadata and
file paths are hashed rather than stored, `meas_date` is recorded only as
present or absent because dates of service are identifiers, and confound
*column names* are recorded while their values are only hashed. Manifests get
committed to public repositories; nothing the adapter writes should make that a
mistake.

Adapters never import their framework at module load, so `import daftar` works
with none of them installed. Every probe is best-effort: a provenance tool that
crashes a four-hour simulation because a framework renamed an attribute has done
far more harm than the missing field was worth.

All nine are verified against live installs by `tests/test_adapters_live.py`.
See `examples/adapter_usage.py` for the pattern for each, and
[INSTALL.md](INSTALL.md) for which environment each needs — they do not all fit
in one.

### Why not Concordia

Every Concordia agent step calls `LanguageModel.sample_text()`, and no major
provider guarantees token-level determinism even with a fixed seed. `replay`
there cannot mean what it means everywhere else, and shipping an adapter whose
replay silently does not replay would undermine the one property this package
sells.

The right design is a different contract — wrap the model, hash every
`(prompt, response)` pair in order, and have `diff` report the first step at
which two runs diverged. That turns Concordia into the strongest argument for
this whole package rather than an awkward fit, because LLM-driven simulation is
the case where nobody can currently audit anything. It comes after the
deterministic adapters have users.

---

## Try it

```bash
python examples/demo_end_to_end.py
```

No frameworks needed. Walks through a clean reproduction, a deliberate parameter
change, a genuine nondeterminism catch, a sweep, a replay plan, and an export.

## Licence

Apache 2.0.
