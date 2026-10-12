"""baghban adapter -- provenance for a Bonsai session.

Bonsai is the odd one out among the frameworks daftar adapts: the experiment
does not run in Python. A rig is started as

    Bonsai.exe rig.bonsai --no-editor --property Animal=M12

and everything that decides what the session recorded lives in a ``.bonsai``
XML file and the NuGet packages it resolves. A generic tracker wrapping that
call sees one string -- the command line -- and can record nothing else,
because it cannot read the format.

What this records instead, via :mod:`baghban`:

* **The effective value of every externalized property.** ``--property`` on the
  command line overrides what is saved in the file, and a dotted name walks
  down through nested workflows by Bonsai's own rules. The command line alone
  does not tell you the value of a property nobody passed; the file alone does
  not tell you the one that was overridden. Only the two together do.
* **A content fingerprint, not a file hash.** Moving a node in the editor
  changes every byte of the file and changes nothing about the protocol.
  ``bonsai.fingerprint`` ignores layout, so two sessions of one protocol share
  it and a real edit breaks it. Both are recorded, because they answer
  different questions.
* **The resolved package versions.** A workflow pulls Bonsai packages through
  ``Bonsai.config``. Nothing in the command line says the rig depended on
  ``Bonsai.Vision 2.8.1``.
* **The include tree.** Includes are separate files that change behaviour, and
  they are hashed as inputs. A researcher asked to list them by hand would
  miss the nested ones.
* **Which files the session will write**, so they can be hashed afterwards --
  including the ones Bonsai renames. A writer with ``Suffix=Timestamp`` does
  not produce ``session.csv``; it produces ``session`` + a timestamp +
  ``.csv``, and hashing the name as written finds nothing.

And one thing no other adapter here does. The others record *what* was run.
This one also records **whether the recording was broken**, before the data is
collected:

    param.bonsai.check.errors            = 1
    param.bonsai.check.SILENT_OVERWRITE  = SelectMany #2 > CsvWriter 'trial.csv' #0

``SILENT_OVERWRITE`` means the CSV about to be hashed holds the last trial
rather than all of them. That is a fact about the data, knowable before the
session starts, and it is the difference between having 200 trials and having
trial 200. It belongs in the run record next to the file hash.

Findings are recorded as **parameters**, not results. daftar's diff separates
fields that could have *caused* a difference from fields that merely record
one, and a broken writer is squarely a cause: a session with
``SILENT_OVERWRITE`` and one without differ in their data because of it.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "baghban"

#: Suffix values that make Bonsai rename the file it writes. ``None`` (the
#: string, as it appears in the XML) and an empty value mean the path is used
#: exactly as written; anything else means Bonsai inserts something before the
#: extension, so the file must be found by pattern rather than by name.
_EXACT_SUFFIXES = {"", "none"}


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see :func:`daftar.adapters.base.probe_import`."""
    return probe_import("baghban")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _key(prefix: str, rest: str) -> str:
    """Join an optional caller prefix to a ``bonsai.``-prefixed field.

    ``baghban.provenance`` already namespaces its keys, so the default is to
    use them unchanged: ``param.bonsai.workflow``. A prefix is only needed when
    one run drives more than one rig, where it separates them:
    ``param.rig2.bonsai.workflow``.
    """
    return f"{prefix}.{rest}" if prefix else rest


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------

def describe_workflow(workflow: Any, run: Run, prefix: str = "", *,
                      properties: Optional[dict] = None,
                      command_line: Any = None,
                      cwd: Any = None,
                      with_check: bool = True,
                      with_inputs: bool = True,
                      **load_options) -> Optional[dict]:
    """Record which Bonsai protocol a session ran.

    ``workflow`` is a path or a loaded ``baghban.Document``; ``properties`` are
    the launcher's ``--property`` values; ``command_line`` is the ``Bonsai.exe``
    arguments instead, from which both are read. A relative workflow path is
    taken against ``cwd``.

    Returns the provenance fields, or ``None`` if the workflow could not be
    read -- in which case the reason is recorded rather than raised. A
    provenance tool that aborts a session because it could not parse a file has
    done more harm than the missing fields are worth.
    """
    fields = safe(lambda: _provenance(workflow, properties, command_line, cwd,
                                      load_options))
    if not fields:
        reason = safe(lambda: _why_unreadable(workflow, properties, command_line,
                                              cwd, load_options),
                      "could not be read")
        run.log_param(_key(prefix, "bonsai.workflow_unreadable"), reason)
        return None

    for field, value in fields.items():
        run.log_param(_key(prefix, field), value)

    if with_inputs:
        _add_workflow_inputs(fields, run, prefix)

    if with_check:
        doc = safe(lambda: _document(workflow, properties, command_line, cwd,
                                     load_options))
        if doc is not None:
            describe_report(safe(lambda: _check(doc)), run, prefix)

    return fields


def describe_launch(command_line: Any, run: Run, prefix: str = "", *,
                    cwd: Any = None, **kwargs) -> Optional[dict]:
    """Record a session from the ``Bonsai.exe`` command line alone.

    The common case for a rig started by a task runner or a batch file: the
    command line names the workflow and carries the session's ``--property``
    values, so nothing else has to be passed.

    The command line itself is recorded too. It is the one thing a generic
    tracker would have captured, and keeping it means the manifest can be
    checked against what was actually run.
    """
    run.log_param(_key(prefix, "bonsai.command_line"),
                  safe(lambda: _render_command(command_line), "<unavailable>"))
    return describe_workflow(None, run, prefix, command_line=command_line,
                             cwd=cwd, **kwargs)


def describe_report(report: Any, run: Run, prefix: str = "") -> None:
    """Record baghban's findings for a workflow.

    Counts by severity, then one field per finding code giving the locations it
    fired at. The code is the key rather than the message, because a code is
    stable across baghban versions and a message is not -- a reworded message
    should not show up in a diff as a changed run.
    """
    if report is None:
        run.log_param(_key(prefix, "bonsai.check"), "<unavailable>")
        return

    findings = safe(lambda: list(report.findings), []) or []
    for severity in ("error", "warning", "info"):
        count = sum(1 for f in findings if safe(lambda f=f: f.severity) == severity)
        run.log_param(_key(prefix, f"bonsai.check.{severity}s"), count)

    by_code: dict[str, list[str]] = {}
    for finding in findings:
        code = safe(lambda f=finding: f.code)
        if not code:
            continue
        by_code.setdefault(code, []).append(
            safe(lambda f=finding: f.location, "?") or "?")
    for code, locations in sorted(by_code.items()):
        run.log_param(_key(prefix, f"bonsai.check.{code}"), "; ".join(locations))

    run.log_param(_key(prefix, "bonsai.check.baghban"),
                  safe(lambda: _baghban_version(), "unknown"))
    run.log_param(_key(prefix, "bonsai.nodes"), safe(lambda: report.n_nodes, "unknown"))


# --------------------------------------------------------------------------
# the data the session wrote
# --------------------------------------------------------------------------

def add_outputs(workflow: Any, run: Run, *, base: Any = None, prefix: str = "",
                properties: Optional[dict] = None, command_line: Any = None,
                cwd: Any = None, **load_options) -> list[str]:
    """Hash the files the workflow said it would write. Call after the session.

    Bonsai opens a relative path against *its* working folder, which is the
    workflow's folder in the editor and the launcher's folder under
    ``--no-editor``. The adapter cannot know which, so ``base`` says; it
    defaults to ``cwd`` when given and the workflow's folder otherwise, and the
    choice is recorded either way.

    A writer whose ``Suffix`` is ``Timestamp`` or ``FileCount`` does not produce
    the path as written -- Bonsai inserts into the name -- so those are matched
    by pattern. Returns the paths that were found and hashed.
    """
    doc = safe(lambda: _document(workflow, properties, command_line, cwd,
                                 load_options))
    if doc is None:
        run.log_param(_key(prefix, "bonsai.outputs_unreadable"),
                      "workflow could not be read")
        return []

    root = _output_base(doc, base, cwd)
    run.log_param(_key(prefix, "bonsai.output_base"), str(root))

    declared = safe(lambda: _outputs(doc), []) or []
    found: list[str] = []
    for index, out in enumerate(declared):
        path = (out or {}).get("path") or ""
        if not path:
            continue
        matches = _resolve_output(root, path, (out or {}).get("suffix", ""))
        if not matches:
            # Declared and absent is a finding, not an error: a disabled branch
            # or an aborted session leaves nothing behind, and saying so is the
            # point.
            run.log_param(_key(prefix, f"bonsai.output.{index}.status"), "not found")
            continue
        for match in matches:
            safe(lambda m=match: run.add_output(m))
            found.append(str(match))
        run.log_param(_key(prefix, f"bonsai.output.{index}.status"),
                      f"{len(matches)} file(s)")
        if (out or {}).get("set_at_run_time"):
            # The path came from data, so what was declared is only a starting
            # value and the match may be unrelated to what the session wrote.
            run.log_param(_key(prefix, f"bonsai.output.{index}.note"),
                          "path assigned at run time; declared value is a starting point")
    return found


@contextlib.contextmanager
def session(workflow: Any, run: Run, prefix: str = "", *, base: Any = None,
            properties: Optional[dict] = None, command_line: Any = None,
            cwd: Any = None, **load_options) -> Iterator[Optional[dict]]:
    """Record the protocol, run the session, then hash what it wrote.

    ::

        with daftar.track("session-M12", seed=1) as run:
            with baghban_adapter.session(
                "rig.bonsai", run,
                command_line=argv, cwd=rig_dir,
            ):
                subprocess.run(argv, cwd=rig_dir, check=True)

    The second half is the half people forget. Recording the protocol before a
    four-hour session and never hashing the files it produced leaves a manifest
    that describes an intention rather than a result, so the outputs are
    collected on the way out -- including when the session raises, since a
    crashed run that wrote three hours of data is exactly the one worth having
    a record of.
    """
    fields = describe_workflow(workflow, run, prefix, properties=properties,
                               command_line=command_line, cwd=cwd, **load_options)
    try:
        yield fields
    finally:
        safe(lambda: add_outputs(workflow, run, base=base, prefix=prefix,
                                 properties=properties, command_line=command_line,
                                 cwd=cwd, **load_options))


# --------------------------------------------------------------------------
# internals -- baghban is imported here, never at module load
# --------------------------------------------------------------------------

def _baghban_version() -> str:
    import baghban
    return baghban.__version__


def _provenance(workflow, properties, command_line, cwd, load_options) -> dict:
    import baghban
    return baghban.provenance(workflow, properties, command_line=command_line,
                              cwd=cwd, **load_options)


def _document(workflow, properties, command_line, cwd, load_options):
    """The loaded workflow, with the launcher's values applied.

    Mirrors what ``provenance`` does internally so that checks and outputs see
    the session's effective values rather than the ones saved in the file.
    """
    import baghban

    props = {}
    if command_line is not None:
        launch = baghban.parse_command_line(command_line)
        props.update(launch.properties)
        if workflow is None:
            workflow = launch.workflow
    props.update(properties or {})
    if workflow is None:
        return None

    if isinstance(workflow, baghban.Document):
        doc = workflow
    else:
        path = Path(workflow)
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        doc = baghban.load(path, **load_options)
    if props:
        baghban.set_properties(doc, props)
    return doc


def _check(doc):
    import baghban
    return baghban.check(doc)


def _outputs(doc) -> list:
    import baghban
    return baghban.outputs(doc)


def _why_unreadable(workflow, properties, command_line, cwd, load_options) -> str:
    """Re-run the load to capture *why* it failed, for the record."""
    try:
        _provenance(workflow, properties, command_line, cwd, load_options)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "could not be read"


def _render_command(command_line) -> str:
    if isinstance(command_line, (str, bytes, os.PathLike)):
        return str(command_line)
    return " ".join(str(a) for a in command_line)


def _add_workflow_inputs(fields: dict, run: Run, prefix: str) -> None:
    """Hash the workflow file and every include that resolved.

    The fingerprint says whether the protocol changed; these say whether the
    bytes did. Keeping both means an edit that only moved nodes is visible as
    exactly that.
    """
    path = fields.get("bonsai.workflow")
    if path:
        safe(lambda: run.add_input(path, name=_key(prefix, "bonsai.workflow")))

    for field, status in fields.items():
        if not field.startswith("bonsai.include.") or status != "ok":
            continue
        include = field[len("bonsai.include."):]
        resolved = _resolve_include(path, include)
        if resolved is not None:
            safe(lambda r=resolved, i=include:
                 run.add_input(r, name=_key(prefix, f"bonsai.include.{i}")))


def _resolve_include(workflow_path: Optional[str], include: str) -> Optional[Path]:
    """Include paths resolve against the folder of the top-level workflow."""
    if not workflow_path:
        return None
    candidate = Path(workflow_path).parent / include
    return candidate if candidate.is_file() else None


def _output_base(doc, base, cwd) -> Path:
    if base is not None:
        return Path(base)
    if cwd is not None:
        return Path(cwd)
    doc_path = safe(lambda: doc.path)
    return Path(doc_path).parent if doc_path else Path.cwd()


def _resolve_output(root: Path, path: str, suffix: Any) -> list[Path]:
    """Files on disk for one declared writer path.

    ``Suffix=None`` means the path exactly; ``Timestamp`` and ``FileCount``
    mean Bonsai inserts into the name before the extension, so the stem is
    matched with anything after it.
    """
    target = Path(path)
    if not target.is_absolute():
        target = root / target

    if str(suffix or "").strip().lower() in _EXACT_SUFFIXES:
        return [target] if target.is_file() else []

    parent = target.parent
    if not parent.is_dir():
        return []
    pattern = f"{target.stem}*{target.suffix}"
    return sorted(p for p in parent.glob(pattern) if p.is_file())


#: The adapter protocol's generic entry point.
describe = describe_workflow
