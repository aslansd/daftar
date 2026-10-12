# Publishing daftar

## The release, start to finish

```bash
conda activate daftar312          # or whichever env has build tooling
pip install build twine
```

**1. Bump the version in two places.** They must match or the build is
confusing later:

```bash
# src/daftar/__version__.py
__version__ = "0.3.4"
# pyproject.toml
version = "0.3.4"
```

**2. Write the changelog entry first, not last.** If you cannot describe the
change in a paragraph that says why it matters, the change is probably not
finished.

**3. Verify.**

```bash
pytest -q                                 # 64 passed, 6 skipped minimum
pytest tests/test_adapters_live.py -v     # in each environment
python examples/demo_end_to_end.py
daftar doctor
```

**4. Build and check.**

```bash
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
```

**5. Test the artifact, not the source tree.** This catches packaging mistakes
the test suite cannot see — a module missing from the wheel, a data file not in
`MANIFEST.in`:

```bash
python -m venv /tmp/check
/tmp/check/bin/pip install dist/daftar-*.whl
/tmp/check/bin/python -c "import daftar; print(daftar.__version__)"
/tmp/check/bin/daftar doctor
```

**6. Upload.**

```bash
python -m twine upload dist/*
git tag v0.3.4 && git push --tags
```

**7. Confirm.** pip caches the index page and PyPI's CDN takes a minute or two,
so an immediate `pip install -U daftar` will often show the old version:

```bash
pip install --no-cache-dir -U daftar
```

---

## Things that have cost time before

**A version number can never be reused**, not even after deleting the release.
If 0.3.4 turns out wrong, ship 0.3.5. Do not delete and retry — 0.1.0 is
permanently burned this way.

**`pip install --no-cache-dir`** when verifying. Without it you will see the
previous version and conclude the upload failed.

**Use an API token, not a password.** PyPI account settings → API tokens. Scope
it to the project after the first upload. Twine 7 prompts for the token
directly, so you do not type `__token__` as a username.

To avoid re-pasting, put it in `~/.pypirc` (mode 600):

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

**TestPyPI is a separate service** with separate accounts and separate tokens. A
PyPI token there returns a bare `403 Forbidden`. It is rarely worth the detour:
`twine check` plus the clean-venv install in step 5 covers what a rehearsal
would tell you.

---

## On the licence

`LICENSE` contains the full Apache-2.0 text and `pyproject.toml` declares
`license = "Apache-2.0"` as an SPDX expression. This is not bookkeeping. The
strategy rests on publishing freely available source code, which has generally
been treated as "information and informational materials". A public repository
with a permissive licence and a licence file present is a different legal object
from a private or licence-gated artifact. Keep it that way.

---

## Before announcing a release publicly

- [ ] `pytest -q` green
- [ ] live adapter suite green in both environments
- [ ] `python examples/demo_end_to_end.py` runs on a fresh clone
- [ ] wheel installed into a clean venv and imported
- [ ] README renders correctly on the live PyPI page
- [ ] **daftar used on your own work, unmodified, for a month**
- [ ] `git tag && git push --tags`

The second-to-last item is the one that matters and the only one still open. It
is the real test of whether the API is unobtrusive enough that you keep using it
when nobody is watching. Do it before the announcement, not after.
