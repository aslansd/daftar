# Roadmap

*Last revised 8 October 2026, against daftar 1.1.1 as published on PyPI.*

This file replaces `daftar-next-steps.md`, `daftar-1.1.0-notes.md`, the earlier
`ROADMAP.md`, and the daftar half of `raftaar-daftar-issue-drafts.md`. Every
number below was read from the published artifact or from the PyPI API, not
from memory.

---

## Where this stands

**1.1.1 on PyPI**, Apache 2.0, **zero runtime dependencies**, **127 tests**,
**ten framework adapters**. First public release 31 August 2026.

| | Status |
|---|---|
| Core: manifests, diff verdicts, sweeps, replay, export, CLI | shipped |
| Ten adapters: Jaxley, Brian2, NetPyNE/NEURON, cpm, MNE-Python, Nilearn, sbi, MeltingPot, gdsfactory, Concordia | shipped, each verified against a live install |
| pytest plugin · notebook staleness detection · run browser · `%%daftar` · `daftar doctor` | shipped |
| Three adapters read upstream's own recorded values (1.1.0) | shipped |
| **pyOpenSci review** | **answered — resubmit after 1 December, see below** |
| **External users** | **none yet — the thing that actually matters** |

```
tests/test_core.py            52
tests/test_features.py        17
tests/test_notebook.py        15
tests/test_adapters_live.py   43   (skips whatever framework is absent)
                             ---
                             127
```

The build list is empty. Everything below is about what the project needs that
is not code.

---

## 1. The upstream finding — closed

The claim behind Fellowship Activity 1 was that scientific packages report that
a fit *converged* without recording whether it merely hit its iteration limit.
Writing daftar's adapters turned that from a suspicion into five filed reports
across five projects:

| project | what happened | outcome |
|---|---|---|
| **Jaxley** [#806](https://github.com/jaxleyverse/jaxley/issues/806) | bug report (`jnp.clip(a_max=)`) | responded, closed |
| **cpm** [#85](https://github.com/DevComPsy/cpm/pull/85) | you opened the PR | **merged** |
| **sbi** [#2014](https://github.com/sbi-dev/sbi/issues/2014) → [#2018](https://github.com/sbi-dev/sbi/pull/2018) | you opened the issue; a maintainer implemented it | **merged** |
| **MNE-Python** [#14335](https://github.com/mne-tools/mne-python/issues/14335) → [#14366](https://github.com/mne-tools/mne-python/pull/14366), [#14370](https://github.com/mne-tools/mne-python/pull/14370) | you opened both PRs | **both merged** |
| **Brian2** [#1874](https://github.com/brian-team/brian2/issues/1874) | open, no maintainer reply | waiting |

Three of the four convergence reports were accepted; one maintainer wrote the
patch himself. That is the finding, and it is now a matter of public record
rather than an assertion.

**On Brian2: leave it.** A silent issue is not a rejected one, and a second
message costs more than it can return. If it is ever answered the sentence gets
better; if not, three of four is already the result.

### What 1.1.0 did with that

Each of the three adapters had been *reconstructing* a value its library now
states, and each reconstruction was subtly wrong against the library's own
definition:

- **cpm** — `len(optimiser.initial_guess)` stops describing what the user asked
  for once `reset()` regenerates the array. Now `optimiser.number_of_starts`.
- **sbi** — the adapter used `<` where sbi's loop uses `<=`, so a fit converging
  exactly at the limit was reported as truncated. Now `summary["converged"]`,
  and **the fallback was corrected too**, not merely demoted.
- **MNE** — `n_iter_ < max_iter` was never valid for Infomax, which signalled
  convergence by assigning `step = max_iter`, so `n_iter_` was the budget either
  way. Now `ICA.converged_`.

Every path records which route it took in a `*_source` field, because a value
the library stated and a value we worked out are different kinds of fact and
only one stays correct when the library changes.

### The process note worth keeping

**1.1.0 shipped a regression that 1.1.1 fixed**: `result.training.converged`
changed from a scalar to a list, and inconsistently — one branch logged a list,
the other a bool. One field, two types, depending on which path fired.

That is precisely the drift this package exists to catch, committed in the
release whose whole point was reading values rather than inventing them. It was
caught by daftar's own test suite, which had been asserting the 1.0.0 contract.
Worth remembering the next time a field's shape seems obviously improvable.

---

## 2. pyOpenSci — the real blocker, and it has a date

[Submission #344](https://github.com/pyOpenSci/software-submission/issues/344)
is filed and the Scope section is complete. It has one response, from
`crhea93`:

> "we require at least three months of sustained package development… I
> encourage you to resubmit once this requirement is met."

**This is not a rejection.** It is a single, objective, waitable criterion.

- First PyPI release: **31 August 2026**
- Three months reached: **≈ 1 December 2026**
- Today: 8 October 2026 — about **38 days** of public history

**Set a calendar reminder for 1 December.** On that date, reply in the existing
thread rather than opening a new submission, and lead with what changed since
August: the release history, the upstream contributions, and anything a real
user reported.

### What this means for the next eight weeks

The earlier roadmap said *"nothing further gets built until there are users."*
That advice was right about **adapters** and wrong as a general rule, because
pyOpenSci's criterion is *sustained development* — and a repository that goes
silent until December resubmits from a weaker position than one with a visible
commit history.

The resolution is not to build features nobody asked for. It is that the work
below is both genuinely needed and visible:

- commits from using daftar on your own work (§3)
- bug fixes from that use
- the Brian2 patch, if they ever reply
- documentation written against real friction rather than imagined friction

That is sustained development. A sixth adapter is not.

---

## 3. Use it on your own work for a month

This has been the open month-7 milestone since before 1.0.0, and it is now also
the cheapest way to satisfy pyOpenSci's criterion. Pick three of your own
repositories — real work, where you would be annoyed if the tool got in the way,
not a demo — and run daftar unmodified for a month.

It is the only real test of whether the API is unobtrusive enough that you keep
using it when nobody is watching. Everything else is a guess about that
question.

**Record what annoys you.** The friction you hit is the only user feedback
available right now, and unlike a survey it costs nothing to collect.

---

## 4. Announcing — after 1 December, not before

The case is much stronger now than when the announcement plan was written, and
stronger again once pyOpenSci accepts. What you can say that is checkable in
under a minute:

- **1.1.1 on PyPI**, `pip install daftar`, Apache 2.0, **zero dependencies**
- **Ten framework adapters**, each verified against a live install
- **127 tests**
- **Five upstream contributions across five projects; four accepted**

And the finding, which is the better opening than any description of the tool:

> Building the adapters surfaced the same gap in four independent frameworks:
> an optimiser stops either because it converged or because it ran out of
> iterations, and MNE, cpm, Brian2 and sbi all reported only that it stopped. A
> fit that never converged was indistinguishable from one that did. Three of the
> four have now changed that, and daftar reads their answer instead of guessing.

### Sequence, when the time comes

1. **Tier 1 — ~15 personal emails**, one specific line each. Write to people
   whose *work* you can name: anyone who opened an issue on Jaxley, Brian2, cpm,
   MNE, NetPyNE or sbi in the past year, and people whose SNUFA or Neuromatch
   talks you remember. Not the six friends who did not reply to the
   questionnaire — they have seen an ask and passed.
2. **Tier 2 — your communities**, one per day maximum: eLife Ambassadors, SNUFA,
   Neuromatch, Mathematics of Neuroscience and AI, Cajal/FENS alumni,
   comp-neuro, IPM. **Message a moderator before posting** in any Slack or
   Discord; most ban tool posts and most say yes when asked first.
3. **Tier 3 — broad**, only with the pyOpenSci badge: Scientific Python
   Discourse, RSE Slack, Mastodon, ReScience.

### Rules

1. Post once. Never bump.
2. Never the same text in two places on the same day — people overlap.
3. Ask moderators first.
4. Reply to everyone, especially the dismissive. One sceptic who engages is
   worth more than fifty stars.
5. Ask for criticism, not stars.
6. **Do not link the discovery questionnaire.** Six emails to friendly contacts
   produced zero responses; that is information, not bad luck. A 27-question form
   is fifteen minutes of unpaid work with no payoff for the person filling it.
   You no longer need it: you have a finished package, so the question is no
   longer "would this be useful?" but "here it is — will you try it?" The second
   is a smaller ask and the answer is behaviour rather than opinion. Keep the
   file for people who already use the tool.

### The reply to expect

Some version of *"isn't this just MLflow?"* It is a fair question:

> Fair question, and MLflow is good — I wouldn't replace it for training runs.
> The difference is what an "experiment" is assumed to be. MLflow is built
> around epochs, metrics over steps, and model artifacts. A Brian2 network has
> none of those: it has an integration method Brian2 chose for you and stores
> nowhere, a `dt`, a schedule, and a seed for the connectivity. daftar records
> those and diffs them.
>
> The other difference is scope: no server, no account, no dependencies, and the
> manifest is plain JSON you can read with `cat`. That's deliberate — it has to
> work on an air-gapped cluster and still be readable in five years if I stop
> maintaining it.

---

## 5. Stop adding adapters

The month-12 gate is **200 GitHub stars and 5 external contributors**. Note what
it does not say: number of adapters. Ten adapters with zero users and twenty
adapters with zero users are the same outcome. One adapter with fifteen users
who would complain if it vanished is a different category of thing.

This matters more here than for most projects. The largest risk in the
feasibility plan was focus — 25 repositories, 150+ courses, several new projects
a month. Expanding the adapter list is exactly what that pattern feels like from
the inside: productive, technical, and a way of not doing the harder thing.

**Write the next adapter when a user asks for it.** That request is also the
first evidence anyone cares.

### If one is ever justified, all five must hold

1. **It records something a generic tracker cannot infer.** An adapter that
   calls `log_params(kwargs)` is not worth the import.
2. **The community feels the pain acutely** — long runs, many parameters,
   results that visibly move between sessions.
3. **You can reach that community.** An adapter you cannot distribute is a hobby.
4. **The framework is stable enough not to rot.** Every adapter is a maintenance
   liability against someone else's release schedule, and three of the ten have
   already broken against a dependency.
5. **It stays inside the envelope** — unregulated, non-dual-use, Python, runs on
   a laptop.

### The only remaining candidate

**FieldTrip / EEGLAB interop**, and it is weak. Much of the EEG world is still
MATLAB. daftar cannot instrument EEGLAB, but it could record the provenance of
data *imported* from it: which `.set` file, which preprocessing was already
applied before Python saw it. Lower value than a native adapter, much cheaper to
build, and still not worth doing before someone asks.

### Explicitly not on the list

- **Generic PyTorch / scikit-learn** — what W&B and MLflow already do well and
  for free. Do not compete on their ground.
- **fMRIPrep** — it already emits good BIDS-derivative provenance. A second
  layer over the preprocessing helps nobody. *(An earlier list dismissed
  "Nilearn / fMRIPrep" together. That was wrong: fMRIPrep's provenance stops
  exactly where nilearn begins, and the nilearn side — confounds, atlas, masker
  settings, GLM design — was as unrecorded as anything else. Nilearn shipped in
  0.6.0.)*
- **MuJoCo, Genesis, Isaac** — robotics customers are defence-adjacent.
- **CFD / HydroGym** — same, plus baselines needing 150,000 GPU-hours.
- **Digital EDA flows** — export-controlled toolchains and foundry NDAs no small
  team gets. *(The gdsfactory adapter does not contradict this: gdsfactory is an
  open-source library for scripting layout geometry in silicon photonics, with an
  open generic PDK, and the adapter records provenance about layout scripts — no
  design capability, no PDK, no foundry data. The line is between recording and
  designing, and between photonics and advanced-node digital logic.)*

---

## 6. What signals something

Roughly ascending:

| Signal | What it means |
|---|---|
| Stars | Almost nothing. People star to bookmark. |
| PyPI downloads | Weak — mostly CI mirrors and bots. |
| Forks | Slightly better; someone wanted to change something. |
| **An issue from a stranger** | **First real evidence someone ran it on their own work.** |
| **A bug report with a traceback** | Better — they hit a real edge case, so real use. |
| **A feature request** | Strongest. They have a use case and want it to fit better. |

**Only build more if someone asks.** A feature request from a stranger is the
signal to resume; nothing else is.

---

## 7. The order

1. **Now → 1 December:** run daftar on three of your own repositories. Fix what
   annoys you. Commit as you go.
2. **If Brian2 replies:** offer the patch. Otherwise leave it.
3. **1 December:** reply in pyOpenSci #344 with the three-month history.
4. **After acceptance:** Tier 1 emails, then Tier 2, one per day.
5. **Then pause** and watch for an issue from a stranger.

Set a reminder for **three months** to check two things: issues opened by people
who are not you, and whether pyOpenSci has assigned an editor. If both are still
zero, that is real information about the wedge and worth more than continuing to
build.

---

## Honest status

The software is finished for its current scope and the upstream finding is
closed — four projects engaged, three changed their code. That is a real result
and better than most tools at this stage can show.

What has not happened is anyone outside the project using it. Ten adapters, a
plugin, a browser and 127 tests do not change that, and neither will an
eleventh adapter. The pyOpenSci clock is the one external constraint with a
date on it; everything else waits on a stranger opening an issue.
