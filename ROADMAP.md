# Adapter roadmap

## First: verifying the three you have

```bash
pip install -e ".[dev]" jaxley cpm-toolbox
pytest tests/test_adapters_live.py -v
```

`tests/test_adapters_live.py` runs real workloads through each adapter and skips
cleanly when a framework is absent. It exists because **adapters fail silently
by design**: `safe()` turns a renamed attribute into `<unavailable>` in the
manifest rather than crashing a four-hour simulation. That is the right
behaviour and it means a broken adapter looks fine until someone reads a
manifest. Only a live run catches it. Re-run this after every framework upgrade,
not only at release.

### Expect MeltingPot to be the hard one

`dm-meltingpot` depends on `dmlab2d`, whose wheel coverage is narrow and
historically weakest on macOS arm64. If it will not install:

- Try Linux, a container, or Colab for that adapter alone.
- If it still fails, **ship without it.** Mark it experimental in the README and
  say the adapter is untested against a live substrate. An honest "untested"
  costs you nothing; a claimed integration that breaks on someone's first
  attempt costs you that person permanently.

The Jaxley and cpm adapters are the ones that must work, because they are the
beachhead.

---

## The uncomfortable part: you probably should not write more adapters yet

Your month-12 gate is **200 GitHub stars and 5 external contributors**. Notice
what it does not say: number of adapters. Adapter count is a vanity metric — it
is legible, it feels like progress, and it is almost entirely decoupled from
whether anyone uses the thing.

Three adapters with zero users and eight adapters with zero users are the same
outcome. One adapter with fifteen users who would complain if it disappeared is
a different category of thing.

This matters more for you than for most people. The largest risk in the
feasibility plan was focus — 25 repositories, 150+ courses, four to six new
projects a month, almost all marked "Supervisor: N/A". Expanding the adapter
list is exactly what that pattern feels like from the inside: productive,
technical, and a way of not doing the harder thing.

**The harder thing, in order:**

1. Run daftar across three of your own repos, unmodified, for a month. Not a
   demo — actual work where you would be annoyed if it got in the way.
2. Open the `cpm` upstream issue and offer the integration. A named integration
   with a freshly published PLOS Comp Biol toolbox buys more distribution than
   six months of adapters, and costs a week.
3. Submit the JOSS paper.
4. Give the SNUFA and Neuromatch talks.
5. Get five people who are not you to use it.

Write the next adapter when a *user* asks for it. That request is also your
first piece of evidence that anyone cares.

---

## The selection rule

When the time comes, an adapter is worth writing only if all five hold:

1. **It records something a generic tracker cannot infer.** The bar from
   `adapters/base.py`: does it capture something the researcher would have
   forgotten? An adapter that calls `log_params(kwargs)` is not worth the
   import.
2. **The community feels the pain acutely.** Long runs, many parameters, results
   that visibly move between sessions.
3. **You can reach that community.** You have standing in comp-neuro through
   eLife, SNUFA, Neuromatch, and Mathematics of Neuroscience. You have none in,
   say, computational chemistry. An adapter you cannot distribute is a hobby.
4. **The framework is stable enough not to rot.** Every adapter is a maintenance
   liability against someone else's release schedule.
5. **It stays inside the envelope.** Unregulated, non-dual-use, Python, runs on
   a laptop.

---

## Ranked candidates

### Tier 1 — same community, same conferences, warm introductions

**1. Brian2** — spiking neural network simulation.

The strongest next pick. Brian2's users *are* the SNUFA audience you already
presented to, and it has a textbook example of hidden state: the code-generation
target. The same model run under `numpy`, `cython`, or `cpp_standalone` produces
subtly different numerics, and nothing in the user's script records which one
ran. Add `defaultclock.dt`, the unit system, `Network` object composition, and
`seed()` semantics, and the adapter has plenty to justify itself.

**2. sbi** — simulation-based inference.

Note the ecosystem adjacency: **sbi and Jaxley both come from the Macke lab.**
Same maintainers, same users, same conferences. If your Jaxley adapter lands
with anyone, sbi is the natural neighbour and the introduction is warm rather
than cold. The hidden state is substantial — density estimator architecture,
`num_simulations`, number of rounds, the proposal at each round, embedding net.
Posterior estimates move under all of it and none of it is in the call.

**3. NEURON / NetPyNE** — the incumbent in biophysical modelling.

The largest install base in the beachhead by a wide margin, and the worst
provenance situation in it. Compiled `.mod` mechanism files are the specific
prize: `nrnivmodl` produces a binary whose provenance is invisible to everything,
and a stale compiled mechanism silently producing old results is a genuinely
common and genuinely painful failure. Hashing the `.mod` sources and the
compiled library would be the single most valuable thing any adapter here does.

Harder to write than Brian2 — hoc, C++, and a much older API surface. Worth it
only once something in Tier 1 has users.

### Tier 2 — bigger audience, one thing to think about

**4. MNE-Python** — EEG/MEG analysis.

Much larger user base than Jaxley, cpm and Brian2 combined, and preprocessing is
a notorious provenance disaster: filter settings, montage, re-referencing, and
above all **manually rejected ICA components**, which are a human decision that
usually exists only in someone's memory. Recording which components were
excluded, by index and by hand, would be immediately valuable.

**An update to the feasibility plan.** That plan deferred fMRI/EEG because human
neuroimaging is health data — GDPR special category, IRB-bound — and that made
it a bad *vertical* when the product was a hosted cloud. That objection largely
dissolves under the current architecture. daftar is local, offline, and never
transmits data; an adapter records provenance about an analysis without ever
touching, storing, or moving a subject's recordings. The regulatory risk lived
in hosting, and you are not hosting.

This is a real consequence of the pivot and it re-opens the one domain where you
have the deepest expertise. Do not act on it yet — it is a Tier 2 item for a
reason — but it should go back on the list.

### Tier 3 — planned, later

**5. Concordia** — the strategic one, and the reason to do it last.

Every agent step calls `LanguageModel.sample_text()`, so `replay` cannot mean
what it means elsewhere. The right contract is different: wrap the model, hash
every `(prompt, response)` pair in order, and have `diff` report the first step
at which two runs diverged. That makes Concordia the strongest demonstration of
the whole thesis — LLM-driven simulation is the case where nobody can currently
audit anything — but it needs the deterministic adapters to have users first,
or it is a clever demo attached to an unused tool.

### Not on the list

- **Generic PyTorch / scikit-learn** — this is what W&B and MLflow already do
  well and for free. Do not compete on their ground.
- **Nilearn / fMRIPrep** — fMRIPrep already emits good BIDS-derivative
  provenance. Adding a second layer helps nobody.
- **MuJoCo, Genesis, Isaac** — robotics customers are defence-adjacent, which
  fails criterion 4 of the original plan.
- **CFD / HydroGym** — same reason, plus their baselines needed 150,000 GPU-hours.
- **EDA / chip design** — fails two criteria simultaneously. Settled earlier.

---

## The thing that probably beats all of them

**A Jupyter and Colab integration, before any fourth adapter.**

A large share of your target users do not run scripts. They live in notebooks,
and notebooks are precisely where provenance dies:

- The git commit is meaningless — the notebook is one file whose cells were
  executed in an order nobody recorded.
- On Colab there is no repository at all, so `code.*` is nearly empty.
- Cells get edited and re-run, so the code that produced a result may no longer
  exist anywhere.

None of that is solved by another adapter. What would help: hash the executed
cell source rather than the file, record execution counts and the actual
execution order, and detect when a cell has been modified since it last ran. A
`%%daftar` magic that captures the cell body verbatim would be genuinely novel —
no existing tool does this well — and it applies to every framework at once
instead of one.

If you write exactly one more thing after verifying the three adapters, make it
this. It multiplies the value of everything already built and it reaches the
notebook-first majority of the people you interviewed.

Second on that same list: a **pytest plugin**, so runs inside a test suite are
tracked automatically. Small, and it makes daftar useful in CI as a regression
check on your own results, which `diff`'s exit code already supports.
