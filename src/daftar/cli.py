"""``daftar`` command line interface."""

from __future__ import annotations

import argparse
import os
import sys

from .diff import compare_many, diff_manifests, render_diff, render_manifest
from .store import RunStore
from .sweep import export_bundle, plan_replay


def _store(args) -> RunStore:
    return RunStore(getattr(args, "dir", None))


def cmd_init(args) -> int:
    path = _store(args).init()
    print(f"Initialised run store at {path}")
    print("Manifests in runs/ are plain JSON and are meant to be committed.")
    return 0


def cmd_list(args) -> int:
    store = _store(args)
    runs = store.list(limit=args.limit, label=args.label)
    if not runs:
        print(f"No runs recorded in {store.dir}")
        return 0
    print(f"{'RUN':<12} {'STATUS':<12} {'STARTED':<21} {'SECS':>8}  LABEL")
    for m in runs:
        print(
            f"{m.run_id:<12} "
            f"{m.get('meta.status', '?'):<12} "
            f"{m.started_at:<21} "
            f"{m.get('cost.wall_clock_s', ''):>8}  "
            f"{m.label}"
        )
    return 0


def cmd_show(args) -> int:
    m = _store(args).load(args.run_id)
    print(f"Run {m.run_id}  {m.label}")
    print()
    print(render_manifest(m, namespaces=args.ns or None))
    return 0


def cmd_diff(args) -> int:
    store = _store(args)
    a, b = store.load(args.a), store.load(args.b)
    d = diff_manifests(a, b, ignore=args.ignore or [])
    print(render_diff(d, show_neutral=args.all))
    # Exit codes let this be used in CI: 0 reproduced, 1 differs.
    return 0 if d.is_reproduction else 1


def cmd_replay(args) -> int:
    m = _store(args).load(args.run_id)
    plan = plan_replay(m, check_current=not args.no_check)
    print(plan.render())
    return 0 if plan.reproducible else 1


def cmd_export(args) -> int:
    store = _store(args)
    m = store.load(args.run_id)
    out = args.output or f"{m.run_id}.zip"
    path = export_bundle(
        m, out, store=store,
        include_inputs=not args.no_inputs,
        include_outputs=not args.no_outputs,
    )
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:.1f} KiB)")
    return 0


def cmd_vary(args) -> int:
    store = _store(args)
    runs = store.list(label=args.label)
    if len(runs) < 2:
        print("Need at least two runs to compare.")
        return 1
    varying = compare_many(runs)
    interesting = {
        k: v for k, v in varying.items()
        if not k.startswith(("meta.", "cost."))
    }
    if not interesting:
        print(f"{len(runs)} runs, and nothing meaningful varies between them.")
        return 0
    print(f"Across {len(runs)} runs, these fields vary:\n")
    width = max(len(k) for k in interesting)
    for k in sorted(interesting):
        vals = sorted(interesting[k])
        shown = ", ".join(vals[:6]) + (" ..." if len(vals) > 6 else "")
        print(f"  {k.ljust(width)}  {len(vals)} values: {shown}")
    return 0


def cmd_reindex(args) -> int:
    n = _store(args).reindex()
    print(f"Reindexed {n} run(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="daftar",
        description="Record and compare what produced each computational result.",
    )
    p.add_argument("--dir", help="run store directory (default: nearest .daftar)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create a run store here")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("list", help="list recorded runs")
    s.add_argument("-n", "--limit", type=int, default=20)
    s.add_argument("-l", "--label")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="print one run's manifest")
    s.add_argument("run_id")
    s.add_argument("--ns", nargs="*", help="only these namespaces, e.g. param result")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("diff", help="compare two runs")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--all", action="store_true", help="include cost and metadata")
    s.add_argument("--ignore", nargs="*", help="field keys or prefixes to skip")
    s.set_defaults(func=cmd_diff)

    s = sub.add_parser("replay", help="what it would take to reproduce a run")
    s.add_argument("run_id")
    s.add_argument("--no-check", action="store_true",
                   help="do not verify current files and environment")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("export", help="write a self-contained archive")
    s.add_argument("run_id")
    s.add_argument("-o", "--output")
    s.add_argument("--no-inputs", action="store_true")
    s.add_argument("--no-outputs", action="store_true")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("vary", help="show what differs across many runs")
    s.add_argument("-l", "--label")
    s.set_defaults(func=cmd_vary)

    s = sub.add_parser("reindex", help="rebuild the index from manifests")
    s.set_defaults(func=cmd_reindex)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # `daftar list | head` closes the pipe early. Python would otherwise
        # print a traceback at shutdown when it flushes stdout, which makes a
        # perfectly normal shell idiom look like a crash. Redirect fd 1 to
        # /dev/null so the interpreter's final flush has somewhere to go.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
