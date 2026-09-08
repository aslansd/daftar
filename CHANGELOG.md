# Changelog

## 0.8.1

**Fixed: a package whose *dependency* is missing was reported as missing itself.**

`pip install netpyne` does not pull NEURON, so `import netpyne` raises
`ImportError: No module named 'neuron'`. `probe_import` matched on the message
and concluded netpyne was not installed — false, and it sends the user to
reinstall a package they already have while `pip` insists the requirement is
already satisfied.

`ImportError.name` says *which* module was missing, so the cases can be told
apart precisely. `daftar doctor` now reports:

```
netpyne     BROKEN  installed, but its dependency 'neuron' is not (pip install neuron)
```

This is the same distinction as 0.3.1's "installed but broken" versus "not
installed", one level deeper. Three outcomes, all of which need a different
response from the user: the package is absent, the package is present but a
dependency is absent, or it imports.

## 0.8.0 — NetPyNE / NEURON

The largest install base in computational neuroscience, and the worst provenance
situation in it. This was Tier 1 item #1 on the roadmap, and it is the last one.

**A stale compiled mechanism silently runs old results.** NEURON mechanisms are
written in NMODL and compiled by `nrnivmodl` into `x86_64/libnrnmech.so`. That
binary is what runs. Edit a `.mod` file, forget to recompile, and the simulation
keeps using the previous mechanism without a word — one of the most common and
most painful failures in the field. The adapter hashes the `.mod` sources *and*
the compiled library and compares modification times:

```
candidate causes (3)
  neuron.compiled_lib_stale  false -> true
  neuron.mod_sources_sha256  46ac78867a1134ff -> bb8885864cd78943
  neuron.stale_warning       (absent) -> a .mod source is NEWER than the
                             compiled library: NEURON is running the previously
                             compiled mechanism. Re-run nrnivmodl.
```

**NetPyNE has four RNG seeds of its own.** `cfg.seeds` holds `conn`, `stim`,
`loc` and `cell`, feeding NEURON's Random123 streams for connectivity,
stimulation, cell positions and cell parameters. Seeding numpy does not touch
them and they default to `1`, so a network can appear reproducible while the
reproducibility is accidental. Unlike Brian2's global `seed()` these live on a
config object rather than a module, so daftar's core cannot reach them —
`run_sim()` sets them from the run's seed and records that it did.

**`hParams.celsius` defaults to 6.3 °C**, a value inherited from the original
squid axon work, and it changes every rate constant in every
temperature-dependent mechanism. Most models that should set it do not.

Also recorded: per-section hashes of `netParams` so a diff says *which* part of
the model changed, the realised connection count (`probability=0.2` draws a new
graph each time and the count is nowhere in the parameters), `nhosts` since
results can differ under different parallel decompositions, NEURON's build hash
which the package version does not carry, and spike-train summaries.

Four live tests against real NetPyNE simulations, including the stale-mechanism
catch and connectivity reproducing under a fixed seed.

**Tier 1 of the roadmap is now empty.** Every adapter that scored well against
the five selection criteria has been built. The constraint from here is not
which framework to support next; it is whether anyone is using the nine that
exist.

## 0.7.0 — gdsfactory

Photonic and analog **layout** provenance. gdsfactory is an open-source Python
library for scripting GDSII geometry, used mostly in silicon photonics research.

**The GDS file has no idea what made it.** A GDSII stream is geometry and
nothing else — no script, no parameters, no library version. It is the artifact
that goes to a foundry, and mask changes are expensive. The adapter records the
content hash of the written GDS alongside everything that produced it, which is
the only way that file stops being an orphan.

**The library stack moves polygons.** Component generators are library code, so
a default bend radius or a router heuristic changing between gdsfactory releases
changes the mask. gdsfactory sits on kfactory sits on KLayout; all three
versions are recorded, because the same script on two machines can produce
different geometry and nothing in the output says so.

**The PDK is the process.** gdsfactory refuses to build without an activated
PDK, but which one and which version decides layers, cross sections and every
device. Recorded by name, version, cell and cross-section counts, and a hash of
the layer map.

Also recorded: design settings readably rather than hashed into a cell name
(gdsfactory derives cell names like `mzi_..._DL20_LY2_LX0p1_Bbend_054caa28`,
which is a fingerprint but not diffable), the netlist hash so a routing change
is distinguishable from a geometry change, port counts and names, bounding box,
and polygon counts per layer from a flattened copy.

### On scope

`ROADMAP.md` has excluded chip design from the start, and this does not reverse
that. The exclusion was about **digital EDA flows** — synthesis, place-and-route,
export-controlled toolchains and commercial PDKs under foundry NDA. gdsfactory
is layout-geometry scripting with an open generic PDK, and the adapter records
provenance *about* layout scripts: it contains no design capability, no PDK, no
foundry data and no device models, in the same way the MNE adapter records
provenance about an analysis without containing patient data.

The line is between recording and designing, and between photonics research and
advanced-node digital logic. The roadmap entry has been rewritten to say so
precisely rather than leaving a blanket exclusion that the code contradicts.

Five live tests against real layout generation, including that identical
parameters reproduce an identical mask hash.

## 0.6.0 — Nilearn

**A correction first.** An earlier version of `ROADMAP.md` listed "Nilearn /
fMRIPrep" under *not on the list*, on the grounds that fMRIPrep already emits
good BIDS-derivative provenance. That was wrong, and the error was conflating a
preprocessing pipeline with the analysis library that runs after it. fMRIPrep's
provenance stops exactly where nilearn begins.

Nilearn objects are scikit-learn estimators, so `get_params()` already exposes
the declared configuration and a generic tracker could capture it. The adapter
earns its place on what `get_params()` cannot reach.

**The confounds.** `masker.fit_transform(img, confounds=df)` takes them as a
*runtime argument*: they are regressed out and then forgotten. Nothing is stored
on the masker and `get_params()` has no `confounds` key. Which columns you chose
— 6 motion regressors, 24 with derivatives and squares, aCompCor, global signal
— determines every connectivity value and every GLM coefficient, and it is
typically a hand-written list comprehension over an fMRIPrep TSV. **This is the
`ica.exclude` of fMRI.** Recorded by name, with a coarse taxonomy (`n_motion`,
`n_compcor`, `n_tissue`, `n_scrub`, `n_cosine`) so a diff can say "the tissue
regressors were dropped" rather than making the reader compare two 24-element
lists.

**The mask that actually resolved.** With `mask_img=None` the mask is *computed
from the data*, so it differs per subject and per `mask_strategy`. The declared
parameter says `None`; only `mask_img_` says how many voxels ran.

**`cov_estimator` resolving to Ledoit-Wolf.** `ConnectivityMeasure` declares
`cov_estimator=None` and resolves it after fitting to Ledoit-Wolf shrinkage,
which pulls the covariance toward the identity. On weakly correlated data that
shrinkage dominates: two analyses differing in a preprocessing choice can
produce *identical* connectivity because both were shrunk to the same place.
`get_params()` reports only the `None`. Found while investigating why a demo
diff showed no effect — the shrinkage was the reason, and it is now recorded.

Also recorded: image geometry and affine hash, atlas region counts, GLM design
matrix columns and shape, and contrast map summaries.

### On subject data

fMRI recordings are health data, so the adapter records structure and never
content. File paths are hashed, confound **column names** are recorded but their
**values** only hashed, and images are summarised by geometry. Same reasoning as
the MNE adapter: manifests get committed to public repositories.

Five live tests against real nilearn maskers, connectivity and GLM.

## 0.5.0 — sbi

sbi has the same shape of problem as the other adapters, in a more acute form:
**almost nothing that determines a posterior is stored on the objects
afterwards.**

**Training hyperparameters vanish.** `train(training_batch_size=200,
learning_rate=5e-4, stop_after_epochs=20, ...)` configures the fit and is then
discarded — nothing on the trainer records what you passed. Two posteriors
trained at different learning rates are indistinguishable. `sbi_adapter.train()`
records them, and flags which values were sbi defaults you never passed, so a
future default change appears as a diff rather than silently moving results.

**Whether training converged.** sbi stops either because validation loss
plateaued or because it hit `max_num_epochs`, and those are entirely different
outcomes. It raises a `UserWarning` for the second and stores nothing you would
notice: `epochs_trained` is in `summary`, but the limit it was compared against
is not. Recorded as `result.training.converged`.

This is now the fourth adapter where the same pattern appears — MNE's ICA
(`n_iter_ == max_iter`), cpm's per-participant convergence, Brian2's
auto-selected integrator, and now sbi. Optimisers stop for two reasons and
report one.

**The proposal is the algorithm.** In multi-round SNPE/SNLE/SNRE, round 1 draws
from the prior and later rounds from the current posterior. sbi records how many
rounds happened but not what each drew from, so an amortised run and a
sequential run with the same simulation budget look alike afterwards. Recorded
per round, with content hashes of the simulations.

**The density estimator is usually a string.** `density_estimator="maf"` becomes
a flow with a depth, width and embedding net chosen by defaults that change
between sbi releases. The adapter records the resolved class, parameter count,
input and condition shapes, and embedding net type.

Also recorded: the prior family and its support bounds, the simulation budget
per round and in total, and `x_o` **by content hash** — observations can be
large and are often measured data that does not belong in a committed manifest.

Five live tests against real NPE inference. sbi pulls PyTorch, which is a large
download but otherwise unconstrained; it shares Environment A.

## 0.4.1

**MNE ICA tests no longer assume scikit-learn is installed.** MNE's default ICA
method, `fastica`, delegates to scikit-learn — an *optional* MNE dependency. So
`import mne` succeeds, `daftar doctor` reports `ok`, and `ica.fit()` raises
`ImportError`. The 0.4.0 tests assumed it was present and failed where it was
not.

The tests now select whichever method works: `fastica` when scikit-learn is
available, MNE's native `infomax` otherwise. Choosing rather than skipping is
deliberate — what these tests exercise is the adapter's recording of exclusions,
convergence and `random_state`, none of which depends on the algorithm. Skipping
would have lost real coverage over an incidental dependency. Verified passing
both with scikit-learn and with it blocked.

**`daftar[mne]` now installs scikit-learn**, since ICA is the adapter's headline
feature and having it fail on first use is a poor introduction.

The adapter itself is unchanged. This is the same distinction that came up with
cpm under SciPy 1.18: **importing is not the same as working**, and an optional
dependency is exactly the gap between the two.

## 0.4.0 — MNE-Python

MNE preprocessing is largely a sequence of **human decisions that are never
written down**. Which channels you marked bad, which ICA components you excluded
after looking at topographies, which epochs were dropped and why: each changes
every downstream number, and each typically survives only in the analyst's
memory.

**`ica.exclude` is the headline.** It is the most consequential unrecorded
decision in EEG/MEG analysis. A reviewer asking "which components did you
remove?" is usually asking a question with no surviving answer six months later.
The adapter records the exclusion list, the count, and the decomposition it
indexes into.

**Whether ICA converged.** `n_iter_ == max_iter` means FastICA hit the iteration
limit and stopped, not that it finished. MNE warns at fit time and stores
nothing you would notice afterwards. Recorded as `result.ica.converged`.

**`random_state=None` is flagged loudly.** ICA without a seed is not
reproducible, and neither are the component indices the exclusions refer to.
The manifest records `none (NOT REPRODUCIBLE)` rather than a bare `None`.

**Filter design, not just the band.** `raw.filter()` updates `info["highpass"]`
and `info["lowpass"]` and then discards `fir_design`, `phase`, `window` and the
transition bandwidths. A zero-phase FIR with a wide transition band and a causal
IIR are different filters, and "1–40 Hz" in a methods section does not
distinguish them. `filter_raw()` records the design and flags which values were
MNE defaults you never passed.

**`epochs.drop_log`.** How many epochs were rejected and for what reason. An
evoked average over 40 surviving epochs is a different quantity from one over
180, and the average alone does not say which you have.

### On subject data

This adapter runs against human neuroimaging recordings, so it records structure
and never content. `subject_info` and file paths are **hashed, not stored**, and
`meas_date` is recorded only as present or absent, because dates of service are
themselves identifiers. Manifests get committed to public repositories; nothing
the adapter writes should make that a mistake.

This is also why an MNE adapter is possible at all. The original feasibility
plan deferred fMRI/EEG because human neuroimaging is health data — but that risk
lived in *hosting* it. daftar is local, offline, and never transmits anything, so
an adapter can record provenance about an analysis without touching a subject's
recordings.

Six live tests against real MNE objects. MNE requires only `scipy>=1.13`, so it
shares an environment with cpm's `scipy<1.18` pin.

## 0.3.3 — documentation and cleanup

No behaviour changes. The four adapters, notebook support and the CLI are all
verified working; this release makes the documentation match.

**Documentation restructured** around the four things a user actually needs to
do, each in its own file rather than scattered across a README:

* `INSTALL.md` — installing daftar, and the per-adapter environment
  requirements. Including the honest part: **the four target frameworks cannot
  currently share one interpreter**, and the two-environment split that works.
* `TESTING.md` — the three test suites, what each covers, and how to read a
  skip. Skips have three distinct meanings and only one of them is ordinary.
* `PUBLISHING.md` — the release sequence, trimmed to the steps that matter.
* `TROUBLESHOOTING.md` — reorganised around what is currently open versus
  already fixed upstream.
* `ROADMAP.md` — updated for the shipped state.

**Removed `diagnose_jaxley.py`.** jaxley 0.14.0 fixed the `jnp.clip(a_max=...)`
incompatibility, so the script has nothing left to diagnose. `daftar doctor`
covers the general case.

**Removed the unused Brian2 log-capture path.** `capture_methods` and
`_MethodCapture` were superseded by `resolve_method`, which reproduces Brian2's
integration-method selection directly and is immune to its caching. Nothing
called them and no test covered them; unused code that reads private framework
internals is a maintenance liability, not an escape hatch.

## 0.3.2

**cpm live tests now skip with an explanation instead of failing** when cpm and
SciPy disagree. cpm 0.25.6 passes `disp=` to `fmin_l_bfgs_b`, which SciPy
removed in 1.18.0, so cpm imports cleanly and raises the moment you fit
anything.

This is the third framework/dependency clash across four adapters — jaxley
against JAX, brian2 against NumPy, cpm against SciPy — so the preflight logic is
now a single `_require_working(name, probe)` helper rather than one bespoke
function per adapter. The rule it encodes: skip for a clash we recognise and can
explain, fail loudly for anything we do not, because an unrecognised error might
be our bug and a silent skip would hide it.

**Runtime checks are reported separately from import checks.** `daftar doctor`
answers "does it import", which is fast and usually enough. It is not the same
question as "does it work": cpm reports `ok` under SciPy 1.18 and then fails on
the first fit. The adapter status test now prints both.

## 0.3.1

**A framework that is installed but broken is no longer reported as absent.**
Every adapter's `is_available()` was a bare `try: import x except: return False`,
so an import that failed for *any* reason looked exactly like a package that was
never installed. The live tests then skipped quietly and nothing said why.

This surfaced with Brian2: `pip install brian2` on Python 3.11 resolves to 2.9.0
(the newest supporting that interpreter), which calls `ndarray.ptp` — removed in
NumPy 2.0. It imports with an `AttributeError`, the tests skipped, and the output
was indistinguishable from Brian2 not being installed at all.

Adapters now expose `availability() -> (status, reason)` with three states:
`available`, `missing`, and **`broken`** carrying the exception. Live tests skip
with that reason attached rather than a generic "not installed".

**New: `daftar doctor`.** Reports the interpreter, the daftar version, and every
adapter's status with the reason for any failure. Exits 1 if anything is broken,
so it is usable in CI. It generalises the ad-hoc `diagnose_jaxley.py`.

```
daftar 0.3.1
python 3.11.16 on Darwin arm64

  brian2      BROKEN  AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
  cpm         ok
  jaxley      ok
  meltingpot  ok
```

`TROUBLESHOOTING.md` documents the Brian2 / NumPy 2 / Python 3.12 trap and why
downgrading NumPy is the wrong fix.

## 0.3.0 — Brian2

Brian2 hides more of what determines the answer than any other framework daftar
supports, which makes it the best argument yet for domain adapters.

**The integration method is chosen for you and then forgotten.** Brian2's
default is not a method but a candidate list, `('exact', 'euler', 'heun')`,
tried in order until one accepts the equations. Which one wins depends on the
equations, so a model that integrated `exact` silently falls back to `euler`
after an edit that makes the system non-linear — changing every result.

Brian2 stores the winner nowhere. `state_updater.method_choice` still holds the
list you never chose from, and the decision appears only in a log line. The
adapter records both:

```
param.group.G.method_choice     ('exact', 'euler', 'heun')   what was permitted
param.group.G.method_resolved   exact                        what actually ran
```

Resolution is done by reproducing Brian2's own selection rather than by reading
its log. That matters: `apply_stateupdater` is cached, so the log line is
emitted only the first time a set of equations is seen in a process. An earlier
draft read the log and recorded the method on run 1 and nothing on run 2 —
making the field appear and disappear between runs, and show up as a spurious
cause in every diff.

**`codegen.target` defaults to `auto`**, which resolves to Cython where a
compiler exists and NumPy where one does not. The same script takes different
code paths on a laptop and on a cluster, and neither the script nor the
preference records which happened. `brian2.codegen_resolved` does.

**Fixed: `track(seed=...)` did not seed Brian2.** `brian2.seed()` delegates to
the current device; seeding numpy does not reach it. Without this,
`connect(p=0.05)` draws a different synaptic graph on every run, so two
otherwise identical runs differ with nothing in the manifest to explain it.
Core seeding now calls it and records `seed.brian2`.

Also recorded: the network schedule (reordering thresholds, synapses and resets
changes results without touching model code), equation hashes with differential
and parameter variable names, threshold/reset/refractory, per-group `dt`,
realised synapse counts and delay ranges, `core.default_float_dtype`, and
monitor output as comparable scalars rather than traces.

Four live tests against a real Brian2 network. Two identical runs now diff to
`identical` including the synapse count, which is the check that proves the
seeding fix.

## 0.2.0 — notebooks

Notebooks are where provenance dies, and until now daftar was no better there
than anything else. A notebook breaks the assumptions the rest of the package
makes: the git commit is close to meaningless because a notebook is one file
whose cells ran in an order nobody recorded; cells get edited and re-run, so the
code that made a figure may exist in no file and no commit; and on Colab there is
no repository at all.

**Two new fields carry the weight.**

`code.cell_sha256` — the source of the cell that ran, hashed *before* execution,
so the record survives you editing the cell afterwards. That is the ordinary way
a notebook result becomes unreproducible, and it is invisible to every
file-based tool.

`code.session_history_sha256` — every cell executed before this one, in
execution order. **A notebook result depends on the whole session, not just the
cell you ran.** Two runs of identical code against different session state are
two different experiments, and nothing on disk distinguishes them. Without this
field such a pair diffs as `nondeterministic`, which is wrong and sends you
hunting a seeding bug that does not exist.

**Nothing to switch on.** `daftar.track()` detects IPython by itself and adds
these fields; existing notebook code gains them with no changes. IPython is
never imported when running as a script.

**`%%daftar` cell magic**, for ergonomics rather than because the automatic path
is second class:

```python
%load_ext daftar

%%daftar montecarlo seed=42
vals = simulate(scale=scale)
run.log_result("mean", float(vals.mean()))
```

The magic captures the cell body verbatim before running it, so the exported
bundle contains `cell.py` — the code that actually ran. On Colab, where the VM
is ephemeral and nothing is committed, that is often the only surviving copy.

**Two fields deliberately kept out of `code.*`.** The execution count increments
on every run, so recording it as code identity would make it a *cause* in every
notebook diff while explaining nothing; it lives in `meta.cell_execution_count`,
which the diff treats as neutral. And the entrypoint is keyed on the cell hash
(`notebook::cell[f629a4b8]`) rather than `In[N]`: stable across re-runs of the
same cell, distinct between cells, and never the useless
`<ipython-input-5-a1b2c3>`.

Colab runtimes also record the accelerator and whether Drive was mounted, since
the same notebook on CPU and on a T4 can give different numerics and the
assignment is not something the user pinned.

**Added** `examples/daftar_in_notebooks.ipynb`, which runs on Colab and
demonstrates the same-cell-different-history case end to end. **Added** 15
notebook tests driven by a real IPython shell rather than mocks.

## 0.1.6

Both fixes here were found by generating manifests for the website demo from
the real adapter code rather than writing example field names by hand.

**Manifest values can no longer contain memory addresses.** Python's default
repr for an arbitrary object is `<Foo object at 0x7f3e...>`, which embeds
`id()` and differs on every run. Any field holding one would make two identical
runs compare unequal and turn every diff into a false positive -- the precise
failure this package exists to detect. `_stringify` now strips addresses, so
the worst case is the uninformative-but-stable `<Foo object>`.

**cpm priors now record the distribution, not the object.** `Value.prior` is
not the string you passed to the constructor: cpm turns
`prior="truncated_normal"` into a *frozen scipy distribution*, whose repr is
`<scipy...truncnorm_gen object at 0x...>`. Manifests now record the
distribution name and its actual parameters:

```
param.model.prior.alpha        truncnorm
param.model.prior_args.alpha   {a: -2.0, b: 2.0, loc: 0.5, scale: 0.25}
```

Those numbers *are* the prior, and two fits with different priors are different
experiments even with identical code and data.

**Fixed: failed cpm fits were counted as converged.** The check was
`bool(status) is True or status == 0`, meant to accept both cpm's
`success: bool` and scipy's integer convention where 0 means success. But in
Python `False == 0` is `True`, so every `success: False` matched the second
clause. `n_converged` therefore always equalled `n_fits`.

That is the field the adapter documentation singles out as the one to watch:
a group mean over 60 participants of whom 7 hit the iteration limit is a
different number from one where all 60 converged. Reporting 60/60 when 7 failed
is worse than reporting nothing. Booleans are now tested before integers --
noting that `isinstance(True, int)` is also `True`, so the order matters.

**Vector-valued parameters keep their values.** cpm allows
`Value(value=[0.1, 0.2, 0.3])`, where `float()` raises and the object is not
iterable -- the numbers live on `.value`, with `__array__` as a second route.
Both are tried before giving up, instead of falling through to a repr.

## 0.1.5

**Fixed: `daftar replay` printed a `pip install` line that could not be run.**
A regression from 0.1.3. The `env.<pkg>.source` fields added in that release
were rendered as if they were packages, so a run using Jaxley from git produced:

```
pip install ... jaxley.source==git+https://github.com/jaxleyverse/jaxley.git#2638cca2665e
```

`jaxley.source` is not a package. Pasting that command fails.

The replay plan now separates versions from origins and emits something
runnable:

```
pip install numpy==2.4.6 "jaxley @ git+https://github.com/jaxleyverse/jaxley.git@2638cca2665e"
# dm_meltingpot: installed from editable:file:///Users/you/meltingpot-main --
#   not fetchable by pip; obtain this source separately
```

Three behaviours, one per kind of install:

* **Index installs** pin by version, as before.
* **VCS installs** carry their URL and resolved commit, because for these the
  version string does not identify the code -- which was the whole reason for
  recording origins in 0.1.3.
* **Editable and local-path installs** are listed as comments, not
  requirements, and now also raise a warning on the plan. Nobody else can fetch
  `file:///Users/you/Downloads/thing`, and pretending a version pin would
  reproduce it is exactly the false confidence this package exists to prevent.

The export bundle README shows origins alongside versions for the same reason.

**MeltingPot adapter verified against a live substrate** for the first time:
`commons_harvest__open` built and stepped, config hash, roles, per-player
returns, Gini, and pinned bot checkpoints all recorded. All three adapters are
now confirmed working against real frameworks.

## 0.1.4

**Fixed a cpm adapter bug that reported the wrong cohort size.** The adapter
described `optimiser.model.data`, but a cpm `Wrapper` holds a *single
participant's* trials as a template -- cpm calls `model.reset(data=participant)`
for each subject in turn. So a 60-participant study was recorded as
`data.n_participants = 1`.

The cohort lives on the optimiser. `optimiser.data` is a `DataFrameGroupBy` and
`optimiser.groups` is the authoritative list of cohort keys. Note that cpm's
attribute named `participants` is *not* the participant list: it is the first
group's DataFrame, kept as a template.

Two related fixes in `describe_data`:

* `len()` on a `DataFrameGroupBy` counts groups, not rows, so trial counts were
  wrong for grouped input. It now reads through `.obj`.
* It now handles DataFrame, DataFrameGroupBy and list-of-participants input,
  which are the three shapes cpm accepts.

A cohort size that is confidently wrong is worse than one that is absent, so
this is pinned by unit tests with a stub rather than left to the live tests,
which only run where cpm is installed.

## 0.1.3

**Fixed the cpm live test.** It built `Value(value=0.1, lower=0.0, upper=1.0)`
and expected it to be fitted. In cpm a parameter is *free* only if it has a
prior -- bounds alone leave it fixed -- so `FminBound` correctly refused to run
a model with no free parameters. The test now supplies a prior, and asserts
`n_parameters` and `n_free_parameters` separately, since those two numbers
differ and only one appears in the model's own repr.

**Jaxley tests now skip on a framework version clash instead of failing.**
Jaxley 0.13.0 calls `jnp.clip(x, a_max=...)`; current JAX removed that argument.
The adapter is fine, and Jaxley `main` already fixed it. A preflight
integration now detects the clash and skips with an explanation. Adapter tests
exist to say whether *the adapter* broke; if another package's incompatibility
shows up as a red failure, the suite stops being trusted.

**Environment capture now records where a package was installed *from*.**
A version string is not an identity. jaxley 0.13.0 on PyPI is broken with
current JAX; jaxley `main` fixes it and still calls itself 0.13.0. A manifest
recording only `env.jaxley = 0.13.0` would call those two environments
identical when one works and one does not -- which is worse than recording
nothing.

daftar now reads PEP 610 `direct_url.json` and records `env.<pkg>.source` for
anything not installed from an index:

```
env.jaxley          0.13.0
env.jaxley.source   git+https://github.com/jaxleyverse/jaxley.git#2638cca2665e
```

Index installs record nothing, which is the correct default: absence means
"from PyPI". Editable and local-path installs are recorded too, so a run made
against a working copy is distinguishable from one made against a release.

**Fixed a real cpm adapter bug: `number_of_starts` was always recorded as 1.**
cpm never retains it. The constructor consumes it to build `initial_guess` with
shape `(number_of_starts, n_free_params)` and then discards it; cpm itself
recovers the count internally as `initial_guess.shape[0]`. The adapter now does
the same.

The `initial_guess_supplied` flag has been removed and replaced with the
**actual guess values**. Whether guesses were user-supplied or drawn at random
is genuinely unrecoverable from the object -- cpm keeps no record of it -- but
the values are recoverable and are strictly more useful. When guesses are drawn
randomly they differ between runs, so `diff` now names
`param.fit.initial_guess` as a candidate cause of a changed fit instead of
reporting the run as nondeterministic with no explanation.

**Fixed the cpm live test again: `approx_grad=True` is required.** cpm forwards
`**kwargs` to scipy's `fmin_l_bfgs_b`, which without it expects the objective to
return `(value, gradient)` rather than a scalar. cpm's own tests and notebooks
all pass it; it is easy to miss because it is not a named argument. The test now
also asserts `param.fit.kwargs.approx_grad` reaches the manifest -- a numerical
setting that changes the answer, lives in an opaque dict, and appears nowhere in
the model definition is exactly what the adapter is for.

**Preflight skips are now narrow.** Only the known Jaxley/JAX `a_max` clash
skips; any other preflight failure lets the test run and fail with a real
traceback. A silent skip on an unrecognised error hides genuine adapter
breakage.

**Fixed an environment-dependent assertion in `test_core.py`.**
`test_unseeded_runs_differ_by_seed_and_that_is_a_real_cause` asserted the causes
were exactly `["seed.value"]`, which holds only when jax is absent. With jax
importable, `apply_seeds` also records `seed.jax_root_key`, which is derived
from the seed and therefore also differs. The test now asserts the *namespace*
-- `seed.value` present, and nothing outside `seed.*` -- which is the real
invariant. Same class of bug as the dirty-tree failure in 0.1.1: a test that
quietly encoded one machine's configuration.

**Live tests print framework versions**, because "the adapter tests passed" only
means something alongside what they passed against.

**Added `TROUBLESHOOTING.md`** covering both failures, the stale-pip-index
problem, and MeltingPot install trouble.

## 0.1.2

**cpm adapter now records restart and parallelism settings.** `number_of_starts`,
`initial_guess`, `parallel`, `cores`, `ppt_identifier` and `libraries` are stored
as plain attributes on the optimiser rather than inside `kwargs`, so the previous
version missed them entirely.

`number_of_starts` is the one that matters. With more than one start and no
explicit `initial_guess`, cpm draws initial guesses at random -- so identical
data, bounds and estimator can converge to different optima across runs. The
manifest now records the restart count and whether the guess was supplied or
drawn, which is the difference between "this fit is irreproducible" and "this
fit is irreproducible and here is why".

**Added `tests/test_adapters_live.py`.** Real workloads against Jaxley, cpm and
MeltingPot, each skipping cleanly when its framework is absent. Adapters read
attributes of fast-moving research code and fail *silently by design* -- a
renamed attribute yields `<unavailable>` in the manifest rather than an
exception. Only a live run notices. Run these after every upgrade of a target
framework, not only at release.

**Added `[all]` extra** installing all three target frameworks.

## 0.1.1

Two bugs found by running the test suite on a machine other than the one it was
written on. Both were environment-dependent, which is a fitting way for a
reproducibility tool to be introduced to its own subject matter.

**Fixed: `code.entrypoint` could record the test runner instead of your code.**
Frame depth was counted with a fixed offset, which is correct when `track()` is
called directly and wrong under pytest, where the stack is deeper — manifests
recorded `.../site-packages/_pytest/python.py` as the entrypoint. The caller is
now found by walking outward until leaving the daftar package and the stdlib
plumbing between it and you. Verified from a script, from a decorated function,
and under pytest.

**Fixed: the test suite depended on the developer's working tree.**
`test_clean_run_has_no_blockers` asserted that a clean run produces a
reproducible replay plan, but never controlled the git state of the working
directory. It passed where the project was not a git repository and failed where
it was and had an untracked `dist/`. The store fixture now chdirs to an isolated
directory, and there is a `git_repo` fixture giving a clean committed repository
for the tests that mean to assert something about git.

The dirty-tree blocker itself was correct and is unchanged: uncommitted edits
cannot be recovered from a manifest, so a run made on a dirty tree is honestly
reported as unreproducible.

**Added tests:** `test_dirty_tree_blocks_replay`,
`test_entrypoint_points_at_the_caller`. 36 passing.

No API changes. Upgrading is optional unless you rely on `code.entrypoint`.

## 0.1.0

First release. Core tracking, manifest format, diff with cause/effect verdicts,
sweeps, replay plans, export bundles, CLI, and adapters for Jaxley, cpm, and
MeltingPot.
