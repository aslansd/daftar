# Changelog

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
