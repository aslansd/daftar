# Troubleshooting

## `pip` cannot see a version you just uploaded

```
ERROR: Could not find a version that satisfies the requirement daftar==0.1.2
       (from versions: 0.1.1)
```

The upload worked. 0.1.2 is on PyPI. pip is reading a **cached copy of the
index page** — and PyPI's CDN also takes a minute or two to propagate a new
release, so trying immediately after upload reliably hits the stale copy.

```bash
pip install --no-cache-dir -U daftar
# or, more thoroughly
pip cache purge && pip install -U daftar
```

Verify what is actually published, bypassing pip entirely:

```bash
curl -s https://pypi.org/pypi/daftar/json | python -c \
  "import json,sys; d=json.load(sys.stdin); print(d['info']['version'], sorted(d['releases']))"
```

### One thing to be careful about

Deleting a release from PyPI does not free the version number. It is burned
permanently — you cannot re-upload `0.1.0` even though nothing occupies it now.
If a release is wrong, publish the next number rather than deleting and retrying.

---

## Jaxley: `TypeError: clip() got an unexpected keyword argument 'a_max'`

**This is not a daftar bug, and it is not really a Jaxley bug either.** It is a
version clash between two packages that are not ours.

Jaxley 0.13.0 (the current PyPI release) contains:

```python
def save_exp(x, max_value: float = 20.0):
    x = jnp.clip(x, a_max=max_value)     # jaxley/solver_gate.py
    return jnp.exp(x)
```

JAX deprecated `jnp.clip`'s `a_min`/`a_max` arguments and later removed them, so
with a current JAX this raises deep inside the HH channel. Jaxley's `main`
branch has already fixed it — the same function now reads `jnp.minimum(x, max_value)`.

Three ways out, best first:

```bash
# 1. Install Jaxley from git, where it is fixed
pip install "jaxley @ git+https://github.com/jaxleyverse/jaxley.git"

# 2. Or pin JAX back to a release that still accepts a_max
pip install "jax<0.7" "jaxlib<0.7"

# 3. Or wait for the next Jaxley release and pin it
```

`tests/test_adapters_live.py` now runs a one-compartment integration as a
preflight and **skips with an explanatory message** instead of failing:

```
SKIPPED [1] Jaxley/JAX version clash: this Jaxley release calls
jnp.clip(a_max=...), which current JAX removed. Not a daftar bug.
```

That distinction matters more than it looks. Adapter tests exist to tell you
whether *your adapter* broke. If a framework's own incompatibility surfaces as a
red failure in your suite, you will eventually stop trusting the suite.

### Worth noticing what this actually is

A patch-level upgrade of a transitive dependency silently broke a simulation
package. That is precisely the failure daftar records, and it is the argument
for the whole project arriving unprompted on your own laptop. In a manifest it
appears as:

```
candidate causes (2)
  env.jax      0.6.2  ->  0.10.2
  env.jaxley   0.12.1 ->  0.13.0
```

Keep this example. It is a far better slide than anything invented, because it
happened while building the tool that would have caught it.

---

## cpm: `ValueError: The model does not contain any free parameters`

Also not a daftar bug — an API subtlety worth knowing.

In cpm, `Parameters.free()` returns only parameters **that have a prior**:

```python
def free(self):
    """Return a dictionary of all parameters with a prior distribution."""
    ...
    if value.prior is not None:
        free.append(key)
```

So this parameter is *fixed*, despite having bounds:

```python
Value(value=0.1, lower=0.0, upper=1.0)                      # not free
```

and this one is free:

```python
Value(value=0.1, lower=0.0, upper=1.0,
      prior="norm", args={"mean": 0.5, "sd": 0.25})         # free
```

`FminBound` refuses to run when nothing is free, which is the error above.

This is a good illustration of the problem the cpm authors describe in their own
paper: models implemented with unstated assumptions. "Has bounds" and "is fitted"
look like the same thing and are not. The adapter records
`model.n_parameters` and `model.n_free_parameters` separately for exactly this
reason — the two numbers differ, and only one appears in the model's own repr.

---

## cpm: `IndexError: invalid index to scalar variable`

Raised from inside scipy's `MemoizeJac`, not from cpm or daftar. The cause:
**`approx_grad=True` is required and you must pass it yourself.**

cpm forwards `**kwargs` straight to scipy:

```python
result = fmin_l_bfgs_b(objective, x0=..., bounds=bounds,
                       args=(model, observed, loss, prior),
                       disp=self.display, **self.kwargs)
```

With no `approx_grad`, scipy defaults to `approx_grad=0`, which means "the
objective returns `(value, gradient)`". cpm's objective returns a scalar, so
scipy tries to subscript a float.

```python
FminBound(model=wrapper, data=data, minimisation=...,
          ppt_identifier="ppt", approx_grad=True)   # required
```

cpm's own tests and example notebooks all pass it. It is easy to miss because
it is not a named constructor argument -- it disappears into `**kwargs`.

Which is the point: a numerical setting that changes the answer, lives in an
opaque dict, and appears nowhere in the model definition. The adapter records
`param.fit.kwargs.approx_grad`, so a manifest shows it even though the model
code never mentions it.

---

## cpm: `number_of_starts` shows as 1 when you passed 2

Fixed in the adapter. cpm does not store `number_of_starts` -- it is consumed in
`__init__` and discarded, surviving only as the first dimension of
`initial_guess`. If you are reading manifests written by an older daftar, that
field was wrong.

This is worth an upstream issue on cpm: the number of restarts, and whether the
initial guesses were supplied or drawn, are not recoverable from a constructed
optimiser. For a toolbox whose paper is about unstated assumptions propagating
between labs, that is a gap worth closing -- and reporting it is a better
opening with the maintainers than a cold pull request.

---

## `pip install` from git appears to succeed but changes nothing

Symptom: the install log ends after "Preparing metadata (pyproject.toml)" with
no "Building wheel" and no "Successfully installed", and the old code is still
imported.

Cause: **the git branch declares the same version as the installed release.**
jaxley `main` is versioned 0.13.0 and so is the PyPI release, so pip decides the
requirement is already satisfied and stops. Nothing is wrong and nothing
happens.

```bash
pip install --force-reinstall --no-deps \
    "jaxley @ git+https://github.com/jaxleyverse/jaxley.git"
```

`--force-reinstall` makes pip replace the existing distribution; `--no-deps`
stops it from rebuilding the entire dependency tree to do it. Confirm with:

```bash
python diagnose_jaxley.py
```

which prints the source of `solver_gate.save_exp` -- `jnp.minimum` means the
fixed build, `jnp.clip(a_max=...)` means the old one is still there.

After this, `env.jaxley.source` in your manifests will show the git URL and
commit, so the two builds are no longer indistinguishable.

---

## Jaxley tests skip even after installing from git

Run with `-rs` to see why pytest skipped:

```bash
pytest tests/test_adapters_live.py -rs
```

The skip reason carries the actual exception. As of the latest change, the
suite **only skips for the known `a_max` version clash**; any other preflight
failure lets the test run and fail loudly with a real traceback.

That is deliberate. A silent skip on an unrecognised error is worse than a red
test, because the adapter could be genuinely broken and nothing would say so.
Skips are for "not our problem, and we know why" — never for "something went
wrong".

---

## MeltingPot will not install

`dm-meltingpot` depends on `dmlab2d`, whose wheel coverage is narrow and weakest
on macOS arm64. Installing from a local clone in editable mode works and is what
was used to verify the adapter (dm-meltingpot 2.4.0, dmlab2d 1.0.0).

If it will not install at all: use Linux, a container, or Colab for that adapter
alone. If it still will not, ship without it and mark the adapter untested. An
honest "untested" costs nothing; a claimed integration that breaks on someone's
first attempt costs you that person.

### One consequence of installing it editable

Every manifest will then contain:

```
env.dm_meltingpot          2.4.0
env.dm_meltingpot.source   editable:file:///Users/you/Downloads/meltingpot-main
```

and `daftar replay` warns that the version string does not identify the code and
nobody else can fetch that path. That warning is correct and worth keeping --
but if you are producing manifests for other people to act on, install a release
build rather than a working copy.

---

## `daftar replay` prints a pip line I cannot run

Fixed in 0.1.5. Versions 0.1.3 and 0.1.4 rendered `env.<pkg>.source` fields as
if they were packages:

```
pip install ... jaxley.source==git+https://github.com/jaxleyverse/jaxley.git#2638cca2665e
```

`jaxley.source` is not a package. Upgrade to 0.1.5, which emits index packages
as version pins, VCS packages as `"name @ url@commit"`, and local or editable
installs as comments plus a warning.

Manifests written by older versions are unaffected -- only the rendering was
wrong, and re-running `daftar replay` on an old run now prints correctly.

---

## Adapter tests pass but a manifest field says `<unavailable>`

Working as designed, and worth understanding. `adapters/base.py` wraps every
probe in `safe()`, so a renamed attribute in a target framework produces
`<unavailable>` rather than an exception. A provenance tool must never crash a
four-hour simulation over a missing metadata field.

The cost is that adapter rot is silent. That is the entire reason
`test_adapters_live.py` exists, and why it should be re-run after every upgrade
of a target framework rather than only at release.
