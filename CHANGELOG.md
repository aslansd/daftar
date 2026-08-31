# Changelog

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
