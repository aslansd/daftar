# Publishing daftar

## Routine release

```bash
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
git tag v0.1.5 && git push --tags
```

Then verify, bypassing pip's cache:

```bash
pip install --no-cache-dir -U daftar
```

The rest of this file is background, kept because each item cost time once.

---

## Why a TestPyPI upload returns 403

**TestPyPI is a completely separate service from PyPI.** Separate database,
separate accounts, separate API tokens. A token issued by `pypi.org` is
meaningless to `test.pypi.org` and is rejected with a bare `403 Forbidden` —
which is exactly what you saw.

In likelihood order:

1. **You used a PyPI token on TestPyPI.** Overwhelmingly the most common cause.
2. **You have no TestPyPI account.** Registering on PyPI does not register you
   on TestPyPI. Sign up separately at
   <https://test.pypi.org/account/register/>.
3. **Email not verified** on the TestPyPI account. Unverified accounts cannot
   upload, and the error is a 403 with no explanation.
4. **The name is taken on TestPyPI.** Its namespace is full of abandoned junk
   and is pruned irregularly. If someone holds `daftar` there you get a 403,
   and it says nothing about real PyPI, where `daftar` is confirmed free.

To rule out 1–3, get a token from
<https://test.pypi.org/manage/account/token/> and retry with `--verbose`:

```bash
python -m twine upload --repository testpypi --verbose dist/*
```

The verbose output distinguishes "bad credentials" from "you do not own this
project name", which the plain error does not.

## The thing worth noticing

Your upload reached **100%** before failing. The bytes crossed the network and
were rejected at the authorization layer. Uploading to the Python package index
from where you are is not blocked. That was the open question, and it is now
answered.

## Recommendation: skip TestPyPI

TestPyPI's purpose is rehearsing a release. You have already done the parts that
matter:

- `python -m build` succeeds with no warnings
- `python -m twine check dist/*` passes both artifacts
- the wheel installs into a clean virtualenv and imports with zero dependencies
- the full suite passes (42 core, plus 7 live adapter tests)

A successful TestPyPI upload would tell you nothing further, and every day spent
debugging it is a day `daftar` remains unclaimed on the index that matters.

```bash
# 1. Account on pypi.org, email verified
# 2. Token: pypi.org -> Account settings -> API tokens -> "Add API token"
#    Scope "Entire account" for the first upload; re-scope to the project after.
python -m twine upload dist/*
```

When prompted, paste the token including the `pypi-` prefix. Twine 7 asks for
the token directly, so you do not need to type `__token__` as a username.

To avoid re-pasting, put it in `~/.pypirc` (mode 600):

```ini
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...
```

Note the two tokens differ: TestPyPI tokens encode `dGVzdC5weXBpLm9yZw` —
base64 for `test.pypi.org`. If yours does not, you are holding the wrong one.

## A version number can never be reused

Not even after deleting the release. If 0.1.0 turns out wrong, ship 0.1.1. Do
not agonise over the first upload; the name is the scarce thing, not the
version.

## Also claim

- GitHub `github.com/aslansd/daftar` (the org name `daftar` was free when checked)
- Update `[project.urls]` in `pyproject.toml` if you use a different path

## Before announcing

- [x] `pytest -q` green
- [x] `python examples/demo_end_to_end.py` runs on a fresh clone
- [x] wheel installed into a clean venv and imported
- [x] all three adapters verified against live frameworks
- [ ] README renders correctly on the live PyPI page
- [ ] **run daftar across three of your own repos, unmodified, for a month**
- [ ] `git tag && git push --tags`

The second-to-last item is the month-7 milestone from the feasibility plan and
the only real test of whether the API is unobtrusive enough that you keep using
it when nobody is watching. Do it before the announcement, not after. Everything
above it is now done.

## On the licence

`LICENSE` now contains the full Apache-2.0 text, and `pyproject.toml` declares
`license = "Apache-2.0"` as an SPDX expression. This is not bookkeeping: the
strategy rests on publishing freely available source code, which has generally
been treated as "information and informational materials". A public repository
with a permissive licence and a licence file present is a different legal object
from a private or licence-gated artifact. Keep it that way.
