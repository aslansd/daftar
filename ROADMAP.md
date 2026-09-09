# Roadmap

## Where this stands

Ten adapters. Notebook and Colab support. 0.9.0 on PyPI, zero runtime
dependencies, Apache 2.0.

| | Status |
|---|---|
| Core: manifests, diff verdicts, sweeps, replay, export, CLI | shipped |
| Jaxley adapter | verified live |
| cpm adapter | verified live |
| MeltingPot adapter | verified live |
| Brian2 adapter | verified live |
| MNE-Python adapter | verified live |
| sbi adapter | verified live |
| Nilearn adapter | verified live |
| gdsfactory adapter | verified live |
| NetPyNE / NEURON adapter | verified live |
| Concordia adapter | shipped (different contract — see below) |
| Notebook / Colab (`%%daftar`, cell + session hashing) | shipped |
| `daftar doctor` | shipped |

```
tests/test_core.py            52 passed
tests/test_notebook.py        15 passed
tests/test_adapters_live.py   36 (skip whatever is absent)
```

The initial version is done. Everything below is about what comes next, and the
first section is the one that matters most.

---

## The uncomfortable part: stop adding adapters

The month-12 gate is **200 GitHub stars and 5 external contributors**. Notice
what it does not say: number of adapters. Adapter count is a vanity metric — it
is legible, it feels like progress, and it is almost entirely decoupled from
whether anyone uses the thing.

Ten adapters with zero users and twenty adapters with zero users are the same
outcome. One adapter with fifteen users who would complain if it disappeared is
a different category of thing.

This matters more here than for most projects. The largest risk in the
feasibility plan was focus — 25 repositories, 150+ courses, four to six new
projects a month, almost all marked "Supervisor: N/A". Expanding the adapter
list is exactly what that pattern feels like from the inside: productive,
technical, and a way of not doing the harder thing.

**The harder thing, in order:**

1. **Run daftar across three of your own repos, unmodified, for a month.** Not a
   demo — actual work where you would be annoyed if it got in the way. This is
   the month-7 milestone and it is still open. It is the only real test of
   whether the API is unobtrusive enough that you keep using it when nobody is
   watching.
2. **Open the cpm upstream issue.** The SciPy `disp` bug breaks cpm for every
   user who upgrades, and the fix is a few lines. Offering the PR turns the
   later provenance conversation into one between contributors rather than a
   cold request. This is your best available opener with any maintainer.
3. **Do the thirty discovery interviews.** Stage 0 never happened. Without them
   you are guessing at which sentence makes a researcher lean forward — and you
   would be guessing in your one good shot at each community.
4. **Submit to pyOpenSci.** Their guidance is explicit that review should come
   *before* the software paper, since feedback often changes the API. Accepted
   packages are fast-tracked through JOSS, so it is the efficient route to the
   month-12 paper rather than a detour from it.
5. **Give the SNUFA and Neuromatch talks.** You have presented at both. A tool
   talk lands differently from a results talk.

Write the next adapter when a *user* asks for it. That request is also your
first evidence that anyone cares.

---

## When the time comes: the selection rule

An adapter is worth writing only if all five hold:

1. **It records something a generic tracker cannot infer.** The bar from
   `adapters/base.py`: does it capture something the researcher would have
   forgotten? An adapter that calls `log_params(kwargs)` is not worth the import.
2. **The community feels the pain acutely.** Long runs, many parameters, results
   that visibly move between sessions.
3. **You can reach that community.** You have standing in comp-neuro through
   eLife, SNUFA, Neuromatch and Mathematics of Neuroscience. An adapter you
   cannot distribute is a hobby.
4. **The framework is stable enough not to rot.** Every adapter is a maintenance
   liability against someone else's release schedule — and three of the nine
   existing ones have already broken against a dependency.
5. **It stays inside the envelope.** Unregulated, non-dual-use, Python, runs on
   a laptop.

---

## Ranked candidates

### Tier 1

**Empty — and that is the point.** Every adapter that scored well against the
five criteria has been built. What remains below is genuinely weaker: narrower
audiences, harder maintenance, or a contract that does not fit.

If you find yourself reaching for one of them, re-read the section above. The
constraint is no longer which framework to support next; it is whether anyone
is using the nine that exist.

### Tier 2

**1. FieldTrip / EEGLAB interop** — reading what other toolboxes produced.

Much of the EEG world is still MATLAB. daftar cannot instrument EEGLAB, but it
can record the provenance of data *imported* from it: which `.set` file, which
preprocessing was already applied before Python saw it. Lower value than a
native adapter and much cheaper to build.

### Tier 3

**Shipped in 0.9.0, on a different contract.** The condition set here was
"after the deterministic adapters have users", and that condition has *not* been
met — there are still no external users. The adapter was built anyway because
the deterministic set is complete and it is the strongest demonstration of the
thesis, but the gate is worth recording as skipped rather than passed.

It does not claim reproducibility. It wraps the language model, hashes every
`(prompt, response)` pair in call order, and reports the first step at which two
runs diverged — distinguishing "asked a different question" (the simulation
state had already diverged) from "same question, different answer" (provider
non-determinism). Those have different remedies and nothing else tells them
apart.

### Not on the list

- **Generic PyTorch / scikit-learn** — what W&B and MLflow already do well and
  for free. Do not compete on their ground.
- **fMRIPrep** — it already emits good BIDS-derivative provenance and a
  boilerplate methods paragraph. A second layer over the *preprocessing* helps
  nobody.

  An earlier version of this list read "Nilearn / fMRIPrep" and dismissed both
  together. **That was wrong**, and the error is worth naming: it conflated a
  preprocessing pipeline with the analysis library that runs after it.
  fMRIPrep's provenance stops exactly where nilearn begins, and everything on
  the nilearn side — which confounds were regressed out, which atlas, which
  masker settings, which GLM design — was as unrecorded as anything else this
  package handles. Nilearn shipped in 0.6.0.
- **MuJoCo, Genesis, Isaac** — robotics customers are defence-adjacent.
- **CFD / HydroGym** — same, plus baselines needing 150,000 GPU-hours.
- **Digital EDA flows** — synthesis, place-and-route, and the commercial PDKs
  that go with them. Export-controlled toolchains, foundry NDAs no small team
  gets, and the most politically contested technology category there is.

  **A distinction worth being precise about.** An earlier version of this list
  said "EDA / chip design" without qualification, and the gdsfactory adapter in
  0.7.0 does not contradict it. gdsfactory is an open-source Python library for
  scripting *layout geometry*, used mostly in silicon photonics research, and it
  ships an open generic PDK. The adapter records provenance *about* layout
  scripts: it contains no design capability, no PDK, no foundry data and no
  device models, in the same way the MNE adapter records provenance about an
  analysis without containing patient data.

  The line is between recording and designing, and between photonics research
  and advanced-node digital logic. It is a real line, but it is narrower than
  the original entry implied, and anyone relying on that entry should read it as
  applying to digital EDA flows specifically.

---

## Non-adapter work worth more than an eighth adapter

**A pytest plugin.** Runs inside a test suite tracked automatically. Small, and
it makes daftar useful in CI as a regression check on your own results, which
`diff`'s exit code already supports.

**Richer notebook capture.** The current implementation hashes the executed cell
and the session history, which is the hard part. Detecting that a cell has been
*modified since it last ran* would be a natural extension and would catch the
single most common way a notebook result becomes unreproducible.

**A run browser.** `daftar list` and `daftar vary` are adequate for tens of runs
and poor for hundreds. A small local HTML view over the manifest store would
cost little and is the obvious first paid feature.
