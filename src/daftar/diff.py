"""Comparing two runs.

This is the feature that has to earn its keep on day one. Everything else in
this package -- provenance, replay, succession -- pays off in months or years,
and tools whose payoff is that distant do not get adopted. ``diff`` answers a
question people ask weekly: *why did this run give a different number than the
one on Tuesday?*

The useful part is not listing changed fields. It is separating fields that
could have *caused* a difference from fields that merely *record* one, and then
saying what that combination implies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manifest import (
    EFFECT_NAMESPACES, Manifest, is_cause, is_effect, is_neutral, namespace_of,
    sort_keys,
)


@dataclass
class FieldChange:
    key: str
    a: str | None
    b: str | None

    @property
    def kind(self) -> str:
        if self.a is None:
            return "added"
        if self.b is None:
            return "removed"
        return "changed"

    @property
    def namespace(self) -> str:
        return namespace_of(self.key)


# Verdicts, in the order the diff logic tests for them.
IDENTICAL = "identical"
NONDETERMINISTIC = "nondeterministic"
EXPLAINED = "explained"
NO_EFFECT = "no_effect"
INCOMPARABLE = "incomparable"

_VERDICT_TEXT = {
    IDENTICAL: (
        "Identical. Same code, same inputs, same environment, same results."
    ),
    NONDETERMINISTIC: (
        "Results differ but nothing that could have caused it does. "
        "This run is not deterministic -- an unseeded RNG, thread scheduling, "
        "or hardware non-determinism. That is a finding, not a glitch."
    ),
    EXPLAINED: (
        "Results differ, and so do things that could explain it. "
        "The candidate causes below are where to look."
    ),
    NO_EFFECT: (
        "Inputs or environment differ but the results did not move. "
        "Useful evidence that the result is robust to those changes."
    ),
    INCOMPARABLE: (
        "These runs recorded different result fields, so their outcomes "
        "cannot be compared directly."
    ),
}


@dataclass
class Diff:
    a: Manifest
    b: Manifest
    changes: list[FieldChange] = field(default_factory=list)
    identical_count: int = 0

    # -- partitions -------------------------------------------------------

    @property
    def causes(self) -> list[FieldChange]:
        return [c for c in self.changes if is_cause(c.key)]

    @property
    def effects(self) -> list[FieldChange]:
        return [c for c in self.changes if is_effect(c.key)]

    @property
    def neutral(self) -> list[FieldChange]:
        return [c for c in self.changes if is_neutral(c.key)]

    @property
    def other(self) -> list[FieldChange]:
        return [
            c for c in self.changes
            if not (is_cause(c.key) or is_effect(c.key) or is_neutral(c.key))
        ]

    # -- interpretation ---------------------------------------------------

    @property
    def verdict(self) -> str:
        if not self.changes:
            return IDENTICAL

        effects, causes = self.effects, self.causes
        structural = [c for c in effects if c.kind != "changed"]

        if not effects and not causes:
            return IDENTICAL  # only cost/meta moved: same run, different clock
        if structural and not any(c.kind == "changed" for c in effects):
            return INCOMPARABLE
        if effects and not causes:
            return NONDETERMINISTIC
        if effects and causes:
            return EXPLAINED
        return NO_EFFECT

    @property
    def verdict_text(self) -> str:
        return _VERDICT_TEXT[self.verdict]

    @property
    def is_reproduction(self) -> bool:
        """True when b reproduces a: no cause and no effect moved."""
        return not self.causes and not self.effects

    def __bool__(self) -> bool:
        return bool(self.changes)


def diff_manifests(a: Manifest, b: Manifest, *, ignore: list[str] | None = None) -> Diff:
    """Compare two manifests field by field.

    ``meta.*`` timestamps and run ids are excluded by default. Every run has a
    different id and start time; reporting those as differences would bury the
    real ones under noise on every single comparison.
    """
    ignore = ignore or []
    always_ignore = {
        "meta.run_id", "meta.started_at", "meta.finished_at", "cost.hostname",
    }

    def skip(key: str) -> bool:
        if key in always_ignore:
            return True
        return any(key == pat or key.startswith(pat.rstrip("*")) for pat in ignore)

    keys = sort_keys(set(a.fields) | set(b.fields))
    changes: list[FieldChange] = []
    same = 0

    for key in keys:
        if skip(key):
            continue
        va, vb = a.fields.get(key), b.fields.get(key)
        if va == vb:
            same += 1
        else:
            changes.append(FieldChange(key=key, a=va, b=vb))

    return Diff(a=a, b=b, changes=changes, identical_count=same)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fmt_change(c: FieldChange, width: int) -> str:
    key = c.key.ljust(width)
    if c.kind == "added":
        return f"  {key}  (absent)  ->  {c.b}"
    if c.kind == "removed":
        return f"  {key}  {c.a}  ->  (absent)"
    return f"  {key}  {c.a}  ->  {c.b}"


def render_diff(d: Diff, *, show_neutral: bool = False) -> str:
    """Human-readable diff. Terse by default; the point is to be scannable."""
    lines: list[str] = []
    a_label = d.a.label or "(unlabelled)"
    b_label = d.b.label or "(unlabelled)"
    lines.append(f"--- {d.a.run_id}  {a_label}  {d.a.started_at}")
    lines.append(f"+++ {d.b.run_id}  {b_label}  {d.b.started_at}")
    lines.append("")

    substantive = [c for c in d.changes if not is_neutral(c.key)]

    if not substantive:
        n_neutral = len(d.changes)
        lines.append(f"No meaningful differences across {d.identical_count} fields.")
        if n_neutral:
            lines.append(
                f"({n_neutral} cost/metadata field(s) differ -- timing and host "
                f"only. Use --all to see them.)"
            )
        lines.append("")
        lines.append(_VERDICT_TEXT[IDENTICAL])
        return "\n".join(lines)

    shown = d.changes if show_neutral else substantive
    hidden = len(d.changes) - len(shown)

    width = max((len(c.key) for c in shown), default=10)

    groups: list[tuple[str, list[FieldChange]]] = [
        ("candidate causes", d.causes),
        ("observed effects", d.effects),
        ("other", d.other),
    ]
    if show_neutral:
        groups.append(("cost and metadata", d.neutral))

    for title, items in groups:
        if not items:
            continue
        lines.append(f"{title} ({len(items)})")
        for c in items:
            lines.append(_fmt_change(c, width))
        lines.append("")

    summary = (
        f"{len(substantive)} meaningful field(s) differ, "
        f"{d.identical_count} identical"
    )
    if hidden:
        summary += f"  (+{hidden} cost/metadata, hidden -- use --all)"
    lines.append(summary)
    lines.append("")
    lines.append(d.verdict_text)
    return "\n".join(lines)


def render_manifest(m: Manifest, *, namespaces: list[str] | None = None) -> str:
    keys = [k for k in m.ordered_keys
            if namespaces is None or namespace_of(k) in namespaces]
    if not keys:
        return "(no fields)"
    width = max(len(k) for k in keys)
    out = []
    current_ns = None
    for k in keys:
        ns = namespace_of(k)
        if ns != current_ns:
            if current_ns is not None:
                out.append("")
            current_ns = ns
        out.append(f"  {k.ljust(width)}  {m.fields[k]}")
    return "\n".join(out)


def compare_many(manifests: list[Manifest]) -> dict[str, set[str]]:
    """Fields that vary across a set of runs -- the sweep's actual axes.

    Handy when you inherit twenty runs and want to know what was varied without
    reading twenty files.
    """
    varying: dict[str, set[str]] = {}
    all_keys: set[str] = set()
    for m in manifests:
        all_keys |= set(m.fields)
    for key in all_keys:
        values = {m.fields.get(key, "(absent)") for m in manifests}
        if len(values) > 1:
            varying[key] = values
    return varying
