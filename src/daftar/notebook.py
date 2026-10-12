"""Provenance for notebooks, where it is hardest and matters most.

A notebook breaks nearly every assumption the rest of this package makes about
where code lives:

* **The git commit is close to meaningless.** A notebook is one file whose cells
  were executed in an order nobody recorded. Two people at the same commit can
  hold entirely different results.
* **The code that produced a result may no longer exist anywhere.** Cells get
  edited and re-run. The version that made the figure in your paper was
  overwritten an hour later and is in no file and no commit.
* **Execution order is not cell order.** ``In[7]``, then ``In[3]``, then
  ``In[12]``. The file reads top to bottom; the session did not.
* **A result depends on every cell executed before it**, not just the one you
  ran. The same cell run in two sessions with different history is two different
  experiments, and nothing on disk distinguishes them.
* **On Colab there is no repository at all**, the VM is ephemeral, and the
  preinstalled package set is not something the user chose.

So the useful unit here is not the file but **the executed cell**, and the
useful context is **the session history that preceded it**. Both are recorded:

``code.cell_sha256``
    Hash of the source of the cell that ran. Survives the cell being edited
    afterwards, which is the common case.
``code.session_history_sha256``
    Hash of every cell executed before this one, in execution order. Two runs of
    identical code against different session state are correctly reported as
    different.

Nothing here needs to be switched on. ``daftar.track()`` detects IPython by
itself and adds these fields. The ``%%daftar`` magic exists for ergonomics, not
because the automatic path is second class.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any

__all__ = [
    "in_notebook", "notebook_context", "load_ipython_extension",
    "current_cell_source", "DaftarMagics",
]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _shell() -> Any | None:
    """The live InteractiveShell, or None outside IPython.

    Deliberately does not import IPython -- only looks for it in
    ``sys.modules``. Importing IPython to ask whether we are in IPython would
    add a second of startup to every script run that is not.
    """
    ipy = sys.modules.get("IPython")
    if ipy is None:
        return None
    try:
        return ipy.get_ipython()
    except Exception:  # pragma: no cover - defensive
        return None


def in_notebook() -> bool:
    return _shell() is not None


def _kernel_flavour(shell: Any) -> str:
    """Distinguish a notebook kernel from a terminal REPL from Colab."""
    if "google.colab" in sys.modules:
        return "colab"
    cls = type(shell).__name__
    if cls == "ZMQInteractiveShell":
        return "jupyter"
    if cls == "TerminalInteractiveShell":
        return "ipython-terminal"
    return cls


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------
# cell source
# --------------------------------------------------------------------------

def current_cell_source(shell: Any | None = None) -> str | None:
    """Source of the cell currently executing.

    IPython keeps raw input in ``history_manager.input_hist_raw`` and mirrors it
    into the user's ``In`` list. The last entry is the cell running right now,
    which is what we want -- not the file, which may not exist, and not the cell
    as it looks later, which may have been edited.
    """
    shell = shell or _shell()
    if shell is None:
        return None
    for source in (
        lambda: list(shell.history_manager.input_hist_raw),
        lambda: list(shell.user_ns.get("In", [])),
    ):
        try:
            hist = source()
        except Exception:
            continue
        if hist:
            return hist[-1]
    return None


def _history(shell: Any) -> list[str]:
    for source in (
        lambda: list(shell.history_manager.input_hist_raw),
        lambda: list(shell.user_ns.get("In", [])),
    ):
        try:
            hist = source()
            if hist:
                return hist
        except Exception:
            continue
    return []


# --------------------------------------------------------------------------
# Colab
# --------------------------------------------------------------------------

def _colab_context() -> dict[str, Any]:
    """Facts about a Colab runtime that change results and vanish on restart."""
    ctx: dict[str, Any] = {"colab": True}

    # Accelerator: the same notebook on CPU and on T4 can give different
    # numerics, and the assignment is not something the user pinned.
    gpu = os.environ.get("COLAB_GPU")
    if gpu:
        ctx["colab_gpu_env"] = gpu
    tpu = os.environ.get("COLAB_TPU_ADDR")
    if tpu:
        ctx["colab_tpu"] = True

    try:  # nvidia-smi is present on GPU runtimes and cheap to query
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            ctx["accelerator"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass

    if "google.colab" in sys.modules:
        ctx["drive_mounted"] = os.path.isdir("/content/drive/MyDrive")
    return ctx


# --------------------------------------------------------------------------
# the context daftar records
# --------------------------------------------------------------------------

def session_meta() -> dict[str, Any]:
    """Session bookkeeping that belongs in ``meta.*``, not ``code.*``.

    The execution count increases every time you run anything, so recording it
    under ``code.*`` would mark it a *cause* and make every notebook diff report
    a change that explains nothing. It is worth keeping -- it locates the cell
    within the session -- but it is metadata, not code identity.
    """
    shell = _shell()
    if shell is None:
        return {}
    try:
        return {"cell_execution_count": int(shell.execution_count)}
    except Exception:
        return {}


def _assigned_names(source: str) -> frozenset[str]:
    """Top-level names a cell binds: assignments, defs, classes, imports.

    Used as a cell's *identity*. Two cells that bind the same names are almost
    always two versions of the same cell -- which is what makes it possible to
    notice that one was edited and re-run.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return frozenset(names)


def session_redefinitions(history: list[str]) -> dict[str, Any]:
    """Find names bound by more than one *distinct* cell body in this session.

    This is the single most common way a notebook result goes stale: you edit a
    cell, re-run it, and everything downstream that already used the old
    definition is now inconsistent with everything that used the new one. The
    notebook on disk shows only the final text, so nothing on screen says which
    of your results came from which version.

    Detecting it needs a notion of "the same cell", and cell identity is not
    something Jupyter records. The names a cell binds are a good proxy: a cell
    that defines ``simulate`` and is later replaced by a different cell that
    also defines ``simulate`` is a redefinition, whether or not the surrounding
    text changed.

    Returns the redefined names and how many distinct bodies bound each.
    """
    seen: dict[str, set[str]] = {}
    for cell in history:
        if not cell.strip():
            continue
        body_hash = _sha(cell)
        for name in _assigned_names(cell):
            seen.setdefault(name, set()).add(body_hash)

    redefined = {name: len(bodies) for name, bodies in seen.items()
                 if len(bodies) > 1}
    return {
        "names": sorted(redefined),
        "counts": redefined,
        "n_redefined": len(redefined),
    }


def notebook_context(cell_source: str | None = None) -> dict[str, Any]:
    """Everything worth recording about the notebook session, or ``{}``.

    ``cell_source`` is supplied by the ``%%daftar`` magic, which knows the cell
    body verbatim. Without it the source is recovered from history, which is
    correct but includes the ``track()`` call itself.
    """
    shell = _shell()
    if shell is None:
        return {}

    ctx: dict[str, Any] = {"notebook": True, "kernel": _kernel_flavour(shell)}


    hist = _history(shell)
    source = cell_source if cell_source is not None else current_cell_source(shell)
    if source is not None:
        ctx["cell_sha256"] = _sha(source)[:16]
        ctx["cell_lines"] = len(source.splitlines())
        # Store the body itself only when the magic handed it to us. Recovering
        # it from history would capture the track() call rather than the work,
        # and a manifest full of its own scaffolding helps nobody.
        if cell_source is not None:
            ctx["cell_source"] = source

    # Everything executed *before* this cell. A result depends on the whole
    # session, and this is the only record of it -- the .ipynb on disk shows
    # cells in file order, not the order they ran.
    if hist:
        prior = hist[:-1] if source is not None and hist[-1] == source else hist
        prior = [c for c in prior if c.strip()]
        ctx["session_n_cells"] = len(prior)
        ctx["session_history_sha256"] = _sha("\n\x00\n".join(prior))[:16]

        # Cells edited and re-run during this session. The result you are about
        # to record was built on top of whichever version happened to run last,
        # and the notebook on disk shows only the final text.
        redefinitions = session_redefinitions(prior)
        ctx["session_n_redefined"] = redefinitions["n_redefined"]
        if redefinitions["names"]:
            ctx["session_redefined"] = redefinitions["names"]
            ctx["session_stale_risk"] = True

    path = _notebook_path(shell)
    if path:
        ctx["notebook_path"] = path

    return ctx


def _notebook_path(shell: Any) -> str | None:
    """Best-effort notebook path. Often unavailable, and that is fine."""
    for probe in (
        lambda: shell.user_ns.get("__vsc_ipynb_file__"),          # VS Code
        lambda: os.environ.get("JPY_SESSION_NAME"),               # Jupyter >=7
        lambda: shell.user_ns.get("__session__"),                 # Colab
    ):
        try:
            value = probe()
        except Exception:
            continue
        if value:
            return str(value)
    return None


def environment_extras() -> dict[str, Any]:
    """Notebook-related fields belonging in the ``env.*`` namespace."""
    if "google.colab" in sys.modules:
        return _colab_context()
    return {}


# --------------------------------------------------------------------------
# %%daftar
# --------------------------------------------------------------------------

def _make_magics():
    from IPython.core.magic import Magics, cell_magic, magics_class

    @magics_class
    class DaftarMagics(Magics):
        """``%%daftar`` -- track a cell, recording its source verbatim."""

        @cell_magic
        def daftar(self, line: str, cell: str):
            """Run a cell inside a tracked run.

            ::

                %%daftar celegans seed=42
                v = simulate(dt=0.025)
                run.log_result("mean_rate_hz", float(v.mean()))

            The cell body is captured *before* execution, so the record survives
            the cell being edited afterwards -- which is the ordinary way a
            notebook result becomes unreproducible.

            A ``run`` object is injected into the namespace, so
            ``run.log_result(...)`` works without importing anything.
            """
            import daftar as _daftar

            label, kwargs = _parse_magic_line(line)
            seed = kwargs.pop("seed", None)
            params = kwargs or None

            with _daftar.track(
                label or "cell",
                params=params,
                seed=int(seed) if seed is not None else None,
                _cell_source=cell,
            ) as run:
                self.shell.user_ns["run"] = run
                self.shell.run_cell(cell, store_history=False)
                self.shell.user_ns["_daftar_last_run_id"] = run.run_id
            print(f"daftar: recorded {run.run_id}")

    return DaftarMagics


def _parse_magic_line(line: str) -> tuple[str, dict[str, str]]:
    """``mylabel seed=42 dt=0.025`` -> ``("mylabel", {"seed": "42", ...})``."""
    label, kwargs = "", {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            kwargs[k] = v
        elif not label:
            label = token
    return label, kwargs


DaftarMagics = None  # populated on extension load


def load_ipython_extension(ipython):  # pragma: no cover - needs a live kernel
    """``%load_ext daftar``"""
    global DaftarMagics
    DaftarMagics = _make_magics()
    ipython.register_magics(DaftarMagics)
    print("daftar: %%daftar cell magic registered")
