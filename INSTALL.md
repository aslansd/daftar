# Installing daftar

## Just the tool

```bash
pip install daftar
```

That is everything. daftar has **zero runtime dependencies** and works on Python
3.10 or newer. It needs no account, no server, and makes no network call.

Verify:

```bash
python -c "import daftar; print(daftar.__version__)"
daftar doctor
```

---

## With the framework adapters

Here the honest answer is more complicated than a single command, and it is
worth understanding why before you start.

**The eight target frameworks cannot currently share one environment.** Each pins
its dependencies loosely, and their dependencies remove APIs on schedules the
frameworks do not track. As of this writing:

| Framework | Constraint | Consequence |
|---|---|---|
| brian2 ≥ 2.10 | requires **Python ≥ 3.12** | earlier brian2 releases call `ndarray.ptp`, removed in NumPy 2.0 |
| cpm-toolbox 0.25.6 | needs **scipy < 1.18** | passes `disp=` to `fmin_l_bfgs_b`, removed in SciPy 1.18.0 |
| dm-meltingpot | needs **dmlab2d**, narrow wheel coverage | weakest on macOS arm64; often needs Python 3.11 or a source build |
| jaxley ≥ 0.14 | fine on current JAX | 0.13.0 was broken; upgrade rather than pin |
| mne ≥ 1.12 | `scipy >= 1.13`, Python ≥ 3.10 | compatible with cpm's `scipy<1.18` pin, so they share an environment |
| gdsfactory ≥ 9 | **Python ≥ 3.12, < 3.15**; pulls KLayout | shares Environment A; the Python floor is strict |
| nilearn ≥ 0.11 | `scikit-learn`, `nibabel`, Python ≥ 3.9 | unconstrained; shares Environment A |
| sbi ≥ 0.23 | pulls **PyTorch**, Python ≥ 3.10 | large download; otherwise unconstrained and shares Environment A |
| mne ICA | **needs `scikit-learn`** | an *optional* MNE dependency: `import mne` succeeds and `ica.fit()` raises `ImportError` |

This is not anyone's fault and it is not unusual. It is the normal condition of
a scientific Python environment, and it is a large part of why daftar records
`env.*` at all.

**Do not try to force all four into one environment.** Use two, and let the test
suite skip whatever is absent in each — both suites go green.

### Environment A — brian2, jaxley, cpm, mne, sbi, nilearn, gdsfactory (Python 3.12)

```bash
conda create --name daftar312 python=3.12
conda activate daftar312

pip install daftar
pip install brian2 jaxley cpm-toolbox mne sbi nilearn gdsfactory
#   sbi pulls PyTorch (~2 GB); gdsfactory pulls KLayout
pip install scikit-learn        # MNE's default ICA (fastica) delegates to it
pip install "scipy<1.18"        # required by cpm-toolbox 0.25.6; mne is fine with it
pip install pytest ipython      # for the test suite and notebook support

daftar doctor
```

Expect:

```
daftar 0.3.3
python 3.12.14 on Darwin arm64

  brian2      ok
  cpm         ok
  jaxley      ok
  meltingpot  -       not installed
  mne         ok
  gdsfactory  ok
  nilearn     ok
  sbi         ok
```

### Environment B — meltingpot (Python 3.11)

```bash
conda create --name daftar311 python=3.11
conda activate daftar311

pip install daftar pytest
pip install dm-meltingpot          # or install from a local clone, see below
```

If the wheel will not install on your platform, clone MeltingPot and install it
editable:

```bash
git clone https://github.com/google-deepmind/meltingpot.git
cd meltingpot && pip install -e .
```

Note the consequence: every manifest will then record

```
env.dm_meltingpot.source  editable:file:///path/to/meltingpot
```

and `daftar replay` will warn that nobody else can fetch that path. That warning
is correct. If you are producing manifests for other people to act on, install a
release build rather than a working copy.

### All extras at once

The extras exist for convenience, but read the constraints above before using
`[all]`:

```bash
pip install "daftar[jaxley]"
pip install "daftar[cpm]"
pip install "daftar[brian2]"
pip install "daftar[mne]"        # pulls scikit-learn too, for ICA
pip install "daftar[sbi]"
pip install "daftar[nilearn]"
pip install "daftar[gdsfactory]"
pip install "daftar[meltingpot]"
pip install "daftar[all]"        # will not resolve cleanly on one interpreter
pip install "daftar[dev]"        # pytest, numpy, ipython
```

---

## For development

```bash
git clone https://github.com/aslansd/daftar.git
cd daftar
pip install -e ".[dev]"
pytest -q
```

See [TESTING.md](TESTING.md) for what the different suites cover and what a
skip means.

---

## Notebooks and Colab

Nothing extra is needed — `daftar.track()` detects IPython by itself. IPython is
never imported when running as a script, so there is no cost to non-notebook
users.

For the `%%daftar` cell magic:

```python
!pip install daftar
%load_ext daftar
```

On Colab this is the whole setup. See `examples/daftar_in_notebooks.ipynb`.

---

## Checking an installation

```bash
daftar doctor
```

Reports the interpreter, the daftar version, and every adapter as `ok`,
`not installed`, or **`BROKEN`** with the exception that caused it. Exits 1 if
anything is broken, so it is usable in CI.

**`ok` means the framework imports.** That is not the same as it working — cpm
under SciPy 1.18 imports cleanly and fails on the first fit. The live adapter
tests run a real workload and will tell you. See
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) if something is `BROKEN`.
