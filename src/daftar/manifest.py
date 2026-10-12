"""The manifest: the single artifact this whole package exists to produce.

Design constraints, in priority order:

1.  A manifest must be readable and understandable by a human with no tools
    installed, three years from now. It is JSON with sorted keys, one fact per
    line. If daftar disappears, the record survives.
2.  It must be diffable by ``git diff`` as well as by us. Hence flat, dotted
    keys rather than deep nesting: a changed solver tolerance shows up as one
    changed line, not a re-indented subtree.
3.  Namespaces carry meaning. ``param.*`` and ``env.*`` are *causes*;
    ``result.*`` are *effects*; ``cost.*`` is neither. The diff engine relies on
    this split, and it is the reason the tool can say "your environment changed
    and your result moved" instead of just listing twelve changed lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1"

#: ``<Foo object at 0x7f3e...>`` -> ``<Foo object>``. CPython reprs embed the
#: id() of the object, which differs on every run and every machine.
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _strip_addresses(text: str) -> str:
    return _ADDRESS_RE.sub("", text)

# Namespaces. Order matters for display.
NS_CODE = "code"      # git commit, dirty state, entrypoint
NS_PARAM = "param"    # anything the researcher chose
NS_SEED = "seed"      # RNG seeds, per library
NS_INPUT = "input"    # content hashes of input files
NS_ENV = "env"        # interpreter, package versions, platform
NS_RESULT = "result"  # scalar outcomes worth comparing
NS_OUTPUT = "output"  # produced files, by content hash
NS_COST = "cost"      # wall clock, memory
NS_META = "meta"      # run id, timestamps, labels, notes

#: Namespaces that can *cause* a different result.
CAUSE_NAMESPACES = (NS_CODE, NS_PARAM, NS_SEED, NS_INPUT, NS_ENV)
#: Namespaces that *are* the result.
EFFECT_NAMESPACES = (NS_RESULT, NS_OUTPUT)
#: Namespaces that are expected to vary and are not evidence of anything.
NEUTRAL_NAMESPACES = (NS_COST, NS_META)

_NS_ORDER = {
    NS_META: 0, NS_CODE: 1, NS_PARAM: 2, NS_SEED: 3,
    NS_INPUT: 4, NS_ENV: 5, NS_RESULT: 6, NS_OUTPUT: 7, NS_COST: 8,
}


def namespace_of(key: str) -> str:
    """``'param.dt'`` -> ``'param'``. Unknown prefixes are their own namespace."""
    return key.split(".", 1)[0]


def is_cause(key: str) -> bool:
    return namespace_of(key) in CAUSE_NAMESPACES


def is_effect(key: str) -> bool:
    return namespace_of(key) in EFFECT_NAMESPACES


def is_neutral(key: str) -> bool:
    return namespace_of(key) in NEUTRAL_NAMESPACES


def sort_keys(keys) -> list[str]:
    """Namespace order first, then alphabetical within a namespace."""
    return sorted(keys, key=lambda k: (_NS_ORDER.get(namespace_of(k), 99), k))


def _stringify(value: Any) -> str:
    """Render a value as a stable string.

    Everything in a manifest is a string. This is deliberate and slightly
    annoying. The alternative -- preserving types -- means ``0.025`` and
    ``0.025000000000000001`` compare unequal across platforms, and a diff tool
    that reports spurious differences gets uninstalled within a week. One
    canonical textual form per value is worth the loss of type fidelity.

    The one hard guarantee: **the output never contains a memory address.**
    Python's default ``repr`` for an arbitrary object is
    ``<Thing object at 0x7f3e...>``, which changes on every run. A field like
    that turns every diff into a false positive and would quietly destroy the
    thing this package is for. Addresses are stripped, leaving ``<Thing
    object>``, which is uninformative but at least honest and stable.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # repr() round-trips exactly in Python 3 and keeps 0.1 as "0.1".
        return repr(value)
    if isinstance(value, (int, str)):
        return _strip_addresses(str(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_stringify(v)}" for k, v in sorted(value.items()))
        return "{" + inner + "}"
    # numpy scalars, Path objects, enums, and anything else with a sane repr.
    for attr in ("item", "__fspath__"):
        if hasattr(value, attr):
            try:
                return _stringify(getattr(value, attr)())
            except Exception:  # pragma: no cover - defensive
                break
    return _strip_addresses(str(value))


@dataclass
class Manifest:
    """A flat, ordered, human-readable record of one run."""

    run_id: str
    fields: dict[str, str] = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        self.fields[key] = _stringify(value)

    def set_many(self, namespace: str, values: dict[str, Any]) -> None:
        for k, v in values.items():
            self.set(f"{namespace}.{k}", v)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.fields.get(key, default)

    # -- views ------------------------------------------------------------

    @property
    def ordered_keys(self) -> list[str]:
        return sort_keys(self.fields)

    def namespace(self, ns: str) -> dict[str, str]:
        return {k: v for k, v in self.fields.items() if namespace_of(k) == ns}

    @property
    def label(self) -> str:
        return self.fields.get("meta.label", "")

    @property
    def started_at(self) -> str:
        return self.fields.get("meta.started_at", "")

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "fields": {k: self.fields[k] for k in self.ordered_keys},
        }

    def to_json(self) -> str:
        # indent=2 and no key re-sorting: we already ordered them meaningfully.
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        schema = str(data.get("schema", "0"))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema {schema!r} is not supported by this version "
                f"of daftar (expected {SCHEMA_VERSION!r})"
            )
        return cls(run_id=data["run_id"], fields=dict(data.get("fields", {})))

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Manifest {self.run_id} ({len(self.fields)} fields)>"
