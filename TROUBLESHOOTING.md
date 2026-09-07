# Troubleshooting

First step, always:

```bash
daftar doctor
```

It reports the interpreter, the daftar version, and every adapter as `ok`,
`not installed`, or **`BROKEN`** with the exception that caused it.

**`ok` means the framework imports. That is not the same as it working** — cpm
under SciPy 1.18 imports cleanly and fails on the first fit. For the runtime
check:

```bash
pytest tests/test_adapters_live.py -rs
```

`-rs` prints skip reasons, which `-q` and `-v` both hide.

---

## The pattern behind most of this page

Three of the four adapters have hit the same shape of problem:

| Framework | Dependency | Removed API | Status |
|---|---|---|---|
| jaxley 0.13.0 | JAX ≥ 0.7 | `jnp.clip(a_max=...)` | **fixed** — upgrade to jaxley ≥ 0.14 |
| brian2 ≤ 2.9 | NumPy ≥ 2.0 | `ndarray.ptp` | **fixed** — brian2 ≥ 2.10, needs Python ≥ 3.12 |
| cpm 0.25.6 | SciPy ≥ 1.18 | `fmin_l_bfgs_b(disp=...)` | **open** — pin `scipy<1.18` |

Research packages pin loosely and their dependencies remove APIs on a schedule
the packages do not track. This is not unusual and it is not anyone's fault. It
is the normal condition of a scientific Python environment, and it is a large
part of why daftar records `env.*` at all — each of these appears in a diff as an
environment change with a matching result change, or as code that simply stops
running.

---

## cpm: `fmin_l_bfgs_b() got an unexpected keyword argument 'disp'`

**Currently open.** cpm 0.25.6 calls:

```python
result = fmin_l_bfgs_b(objective, x0=..., bounds=bounds,
                       args=(model, observed, loss, prior),
                       disp=self.display, **self.kwargs)
```

SciPy deprecated `disp` and `iprint` for L-BFGS-B and **removed them in 1.18.0**,
so cpm imports cleanly and raises the moment you fit anything. On SciPy 1.17 you
get the warning instead:

```
DeprecationWarning: The `disp` and `iprint` options of the L-BFGS-B solver
are deprecated and will be removed in SciPy 1.18.0
```

**Fix:**

```bash
pip install "scipy<1.18"
```

Worth reporting upstream — it breaks cpm for every user who upgrades SciPy and
the fix is a few lines: drop `disp`, or pass it only when
`scipy.__version__ < 1.18`.

The live tests skip with this explanation rather than failing.

## cpm: `ValueError: The model does not contain any free parameters`

Not a bug — an API subtlety. `Parameters.free()` returns only parameters **that
have a prior**. So this is *fixed*, despite having bounds:

```python
Value(value=0.1, lower=0.0, upper=1.0)                      # not free
```

and this is free:

```python
Value(value=0.1, lower=0.0, upper=1.0,
      prior="norm", args={"mean": 0.5, "sd": 0.25})         # free
```

`FminBound` refuses to run when nothing is free. The adapter records
`model.n_parameters` and `model.n_free_parameters` separately for exactly this
reason: the two numbers differ, and only one appears in the model's own repr.

## cpm: `IndexError: invalid index to scalar variable`

`approx_grad=True` is required and you must pass it yourself. cpm forwards
`**kwargs` to SciPy, which without it defaults to `approx_grad=0` — meaning "the
objective returns `(value, gradient)`". cpm's returns a scalar.

```python
FminBound(model=wrapper, data=data, minimisation=...,
          ppt_identifier="ppt", approx_grad=True)   # required
```

It is easy to miss because it is not a named constructor argument. The adapter
records `param.fit.kwargs.approx_grad` so a manifest shows it even though the
model code never mentions it.

---

## MNE: `The sklearn package is required to use method='fastica'`

`import mne` succeeds; `ica.fit()` raises. scikit-learn is an **optional** MNE
dependency and MNE's default ICA method delegates to it.

```bash
pip install scikit-learn
```

Or use MNE's native implementation, which needs nothing extra:

```python
mne.preprocessing.ICA(n_components=20, method="infomax", random_state=97)
```

`daftar doctor` reports mne as `ok` here, correctly — the package imports. This
is the same distinction as cpm under SciPy 1.18: **importing is not the same as
working**, and an optional dependency is exactly the gap between them. The live
tests pick whichever ICA method is available rather than skipping, since what
they exercise — recording exclusions, convergence and `random_state` — does not
depend on which algorithm ran.

`pip install "daftar[mne]"` installs scikit-learn alongside mne for this reason.

---

## Brian2: `type object 'numpy.ndarray' has no attribute 'ptp'`

**Resolved by upgrading.** A three-way version trap:

| | |
|---|---|
| NumPy 2.0 removed `ndarray.ptp` | you have numpy 2.x |
| brian2 ≤ 2.9 still calls it | so it cannot import under numpy 2 |
| brian2 ≥ 2.10 fixed it, and **requires Python ≥ 3.12** | if your env is 3.11, pip resolves to 2.9 |

On Python 3.11 there is no brian2 version compatible with NumPy 2.x. Use a
Python 3.12 environment — see [INSTALL.md](INSTALL.md).

**Do not downgrade NumPy instead.** It would fix Brian2 and break jaxley, cpm
and pandas. Pinning NumPy back to satisfy one framework is how an environment
becomes unreproducible in the way this package exists to detect.

---

## Jaxley: `clip() got an unexpected keyword argument 'a_max'`

**Fixed upstream.** jaxley 0.13.0 called `jnp.clip(x, a_max=...)`, which current
JAX removed. jaxley ≥ 0.14 uses `jnp.minimum`.

```bash
pip install -U jaxley       # 0.14.0 or newer
```

If you previously worked around this by installing from git, note that `main`
also reported version 0.13.0, so `pip install "jaxley @ git+..."` was a **no-op**
unless you passed `--force-reinstall`. Reinstall from PyPI to get a clean
version string:

```bash
pip install --force-reinstall --no-deps jaxley
```

---

## MeltingPot will not install

`dm-meltingpot` depends on `dmlab2d`, whose wheel coverage is narrow and weakest
on macOS arm64. Installing from a local clone in editable mode works and is how
the adapter was verified.

If it will not install at all, use Linux, a container, or Colab for that adapter
alone. If it still will not, ship without it and mark the adapter untested. An
honest "untested" costs nothing; a claimed integration that breaks on someone's
first attempt costs you that person.

### One consequence of installing it editable

Every manifest will contain:

```
env.dm_meltingpot          2.4.0
env.dm_meltingpot.source   editable:file:///Users/you/meltingpot-main
```

and `daftar replay` warns that the version string does not identify the code and
nobody else can fetch that path. That warning is correct. If you are producing
manifests for other people to act on, install a release build rather than a
working copy.

---

## `pip install` from git appears to succeed but changes nothing

The install log ends after "Preparing metadata" with no "Building wheel", and
the old code is still imported.

Cause: **the git branch declares the same version as the installed release**, so
pip decides the requirement is already satisfied and stops.

```bash
pip install --force-reinstall --no-deps "pkg @ git+https://github.com/org/pkg.git"
```

`--force-reinstall` makes pip replace the distribution; `--no-deps` stops it
rebuilding the whole dependency tree to do so.

---

## `pip` cannot see a version you just uploaded

```
ERROR: Could not find a version that satisfies the requirement daftar==0.3.3
```

pip is reading a cached index page, and PyPI's CDN takes a minute or two to
propagate.

```bash
pip install --no-cache-dir -U daftar
```

Check what is actually published, bypassing pip:

```bash
curl -s https://pypi.org/pypi/daftar/json | python -c \
  "import json,sys; d=json.load(sys.stdin); print(d['info']['version'])"
```

---

## Adapter tests pass but a manifest field says `<unavailable>`

Working as designed. `adapters/base.py` wraps every probe in `safe()`, so a
renamed attribute in a target framework produces `<unavailable>` rather than an
exception. A provenance tool must never crash a four-hour simulation over a
missing metadata field.

The cost is that adapter rot is silent. That is the entire reason
`tests/test_adapters_live.py` exists, and why it should be re-run after every
upgrade of a target framework rather than only at release.

---

## A manifest field contains a memory address

It should not — `manifest._stringify` strips them, so the worst case is the
uninformative but stable `<Foo object>`. If you see `0x7f3e...` in a manifest,
that is a bug worth reporting: an address changes on every run and would make
two identical runs compare unequal, turning every diff into a false positive.
