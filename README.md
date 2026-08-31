# daftar

**دفتر** — *ledger, register, record-book.* The same word in Persian, Turkish
(*defter*), Azerbaijani, Arabic, Urdu, and Hindi.

Record what produced each computational result — the environment, code version,
parameters, random seeds, and input files — so two runs can be compared and any
past run can be rebuilt.

Experiment trackers solved this for deep learning, where an experiment is
`model.fit()`. They do not help when the experiment is a Hodgkin–Huxley
simulation of 302 neurons, a hierarchical fit across 60 participants, or a
200-episode multi-agent sweep. Those runs have no epochs, no loss curves, and no
checkpoints. They have parameter grids, solver tolerances, random seeds, and
derived quantities.

**No dependencies. Runs offline. No account, no server, no network call.**

```bash
pip install daftar
```

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

`daftar` inverts it. `diff` answers a question people ask weekly — *why is
this number different from Tuesday's?* — and provenance arrives as a free side
effect of a tool they already wanted. Reproducibility is the by-product, not
the pitch.

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
| `code.*` | git commit, branch, dirty flag, **hash of uncommitted changes**, entrypoint, argv |
| `param.*` | everything you chose, including defaults you never passed |
| `seed.*` | seeds **applied** to `random`, numpy, torch; the JAX root key |
| `input.*` | sha256 and size of every declared input file or directory |
| `env.*` | interpreter, OS, and versions of packages the run actually imported |
| `result.*` | scalar outcomes worth comparing |
| `output.*` | sha256 of produced files |
| `cost.*` | wall clock, host, cpu count |

Two of these are less obvious than they look.

**Seeds are applied, not merely recorded.** A tool that only writes down the
seed is close to useless: if the code seeded itself from the clock, recording
that fact tells you the run is irreproducible but does nothing to fix it.
`track(seed=...)` sets every RNG it can reach, then records what it set. If you
pass no seed, one is generated, applied, and recorded — an accidental seed that
is written down is reproducible; a deliberate one that isn't, is not.

**Dirty working trees get their diff hashed.** "Dirty" alone tells you there
were uncommitted edits but not whether they were the *same* edits. Hashing the
diff means two dirty runs can still be proven identical, which is the common
case during a debugging session.

---

## The four commands

```bash
daftar list                    # what has been run
daftar show r-4f21ab           # one run's full manifest
daftar diff r-4f21ab r-88c07e  # what changed, and whether it mattered
daftar vary -l my-sweep        # which fields differ across many runs
daftar replay r-4f21ab         # what it would take to reproduce this
daftar export r-4f21ab -o run.zip
```

`diff` exits 0 if the second run reproduces the first and 1 otherwise, so it
works in CI as a regression check on your own results.

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
enough to notice. `daftar` reports it as a finding rather than a glitch.

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

## Replay

`daftar replay` prints a plan, and deliberately does **not** execute
anything. Re-running arbitrary recorded code would mean this package executes
whatever a manifest tells it to, and it still could not restore your CUDA
driver. What it does honestly is state the target state, check the current state
against it, and list every discrepancy:

```
Replay plan for r-4f21ab

  entrypoint   sim/celegans_hh.py::run_network
  commit       9c1d0ae
  seed         42

  BLOCKERS -- this run cannot be reproduced as recorded:
    - input file changed since the run: data/connectome.csv
      (was a7f39b21, now 3e0c77af)
```

---

## Export

`daftar export` writes a zip containing `README.md`, `manifest.json`,
`fields.tsv`, and the referenced input and output files. The README is generated
in plain English at the archive root, so a successor learns what they are
looking at without installing anything. A bundle that needs our tool to be
understood defeats its own purpose.

---

## Framework adapters

The core tracks any Python function. An adapter earns its existence only by
knowing something a generic tracker cannot infer.

| Adapter | Records what you'd otherwise lose |
|---|---|
| `jaxley` | morphology (compartments, branches, channels, synapses), `jx.integrate` defaults you never passed, `jax_enable_x64`, backend |
| `cpm` | parameter **bounds and priors**, estimator and its scipy method/tolerance, per-participant convergence counts, cohort hash |
| `meltingpot` | resolved substrate ConfigDict hash, roles, episode-length cap, pinned bot checkpoints, per-player returns and Gini |

```python
from daftar.adapters import jaxley as jxa

with daftar.track("hh-cell", seed=0) as run:
    v = jxa.integrate(cell, run, t_max=10.0, delta_t=0.025)
```

Adapters never import their framework at module load, so `import daftar`
works with none of them installed. Every probe is best-effort: a provenance tool
that crashes a four-hour simulation because a framework renamed an attribute has
done far more harm than the missing field was worth.

See `examples/adapter_usage.py` for the full pattern for each.

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

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Licence

Apache 2.0.
