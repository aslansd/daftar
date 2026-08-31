"""Where runs live.

Two representations, deliberately redundant:

* ``.daftar/runs/<id>.json`` -- the truth. Plain files, one per run,
  readable and greppable without this package, safe to commit to git.
* ``.daftar/index.db`` -- a SQLite cache for listing and querying.

If the index is deleted or corrupted it is rebuilt from the JSON files. If the
JSON files are deleted, the run is gone. Keeping the authoritative copy in the
format a human can read -- rather than the format a machine prefers -- is the
whole point; a database you cannot open in three years is not a record.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterator

from .manifest import Manifest

DEFAULT_DIRNAME = ".daftar"
ENV_VAR = "DAFTAR_DIR"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    label       TEXT,
    started_at  TEXT,
    status      TEXT,
    duration_s  TEXT,
    commit_hash TEXT,
    dirty       TEXT
);
CREATE TABLE IF NOT EXISTS fields (
    run_id TEXT,
    key    TEXT,
    value  TEXT,
    PRIMARY KEY (run_id, key)
);
CREATE INDEX IF NOT EXISTS idx_fields_key ON fields(key);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""


def find_store_dir(start: str | os.PathLike | None = None) -> Path:
    """Locate the store: env var, then nearest ancestor ``.daftar``, then cwd.

    Walking upwards means a script in ``project/sims/deep/run.py`` writes to
    ``project/.daftar`` rather than creating a fourth store in a subfolder --
    the same reason git looks upward for ``.git``.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / DEFAULT_DIRNAME).is_dir():
            return candidate / DEFAULT_DIRNAME
    return here / DEFAULT_DIRNAME


class RunStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.dir = Path(path).resolve() if path else find_store_dir()
        self.runs_dir = self.dir / "runs"
        self.bundles_dir = self.dir / "bundles"
        self.db_path = self.dir / "index.db"
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------

    def init(self) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.dir / ".gitignore"
        if not gitignore.exists():
            # Manifests are meant to be committed. Everything else is cache.
            gitignore.write_text(
                "# Manifests in runs/ are the record -- commit them.\n"
                "index.db\n"
                "bundles/\n"
            )
        self._connect().executescript(_SCHEMA)
        self._connect().commit()
        return self.dir

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- writing ----------------------------------------------------------

    def save(self, manifest: Manifest) -> Path:
        self.init()
        path = self.runs_dir / f"{manifest.run_id}.json"
        # Write-then-rename: a run interrupted mid-write leaves the previous
        # manifest intact rather than a truncated file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(manifest.to_json(), encoding="utf-8")
        tmp.replace(path)
        self._index(manifest)
        return path

    def _index(self, m: Manifest) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, label, started_at, status, duration_s, commit_hash, dirty) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                m.run_id,
                m.get("meta.label", ""),
                m.get("meta.started_at", ""),
                m.get("meta.status", "unknown"),
                m.get("cost.wall_clock_s", ""),
                m.get("code.commit_short", ""),
                m.get("code.dirty", ""),
            ),
        )
        conn.execute("DELETE FROM fields WHERE run_id = ?", (m.run_id,))
        conn.executemany(
            "INSERT INTO fields (run_id, key, value) VALUES (?,?,?)",
            [(m.run_id, k, v) for k, v in m.fields.items()],
        )
        conn.commit()

    # -- reading ----------------------------------------------------------

    def load(self, run_id: str) -> Manifest:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            match = self.resolve(run_id)
            if match is None:
                raise KeyError(f"no run matching {run_id!r} in {self.dir}")
            path = self.runs_dir / f"{match}.json"
        return Manifest.from_json(path.read_text(encoding="utf-8"))

    def resolve(self, prefix: str) -> str | None:
        """Accept an unambiguous id prefix, the way git accepts short SHAs."""
        matches = [r for r in self.list_ids() if r.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"{prefix!r} is ambiguous: matches {', '.join(sorted(matches)[:5])}"
            )
        return None

    def list_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(p.stem for p in self.runs_dir.glob("*.json"))

    def iter_manifests(self) -> Iterator[Manifest]:
        for run_id in self.list_ids():
            try:
                yield self.load(run_id)
            except Exception:
                continue

    def list(self, limit: int | None = None, label: str | None = None) -> list[Manifest]:
        items = list(self.iter_manifests())
        if label:
            items = [m for m in items if m.label == label]
        items.sort(key=lambda m: m.started_at, reverse=True)
        return items[:limit] if limit else items

    def reindex(self) -> int:
        self.init()
        conn = self._connect()
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM fields")
        conn.commit()
        n = 0
        for m in self.iter_manifests():
            self._index(m)
            n += 1
        return n

    def query(self, key: str, value: str | None = None) -> list[str]:
        """Run ids having ``key`` (optionally equal to ``value``)."""
        conn = self._connect()
        if value is None:
            rows = conn.execute("SELECT run_id FROM fields WHERE key = ?", (key,))
        else:
            rows = conn.execute(
                "SELECT run_id FROM fields WHERE key = ? AND value = ?", (key, value)
            )
        return sorted(r["run_id"] for r in rows)
