"""daftar -- record what produced each computational result.

    import daftar

    with daftar.track("celegans", params={"dt": 0.025}, seed=42) as run:
        run.add_input("data/connectome.csv")
        v = simulate(dt=0.025)
        run.log_result("mean_rate_hz", float(v.mean()))

Then, from a shell::

    daftar list
    daftar diff r-4f21ab r-88c07e
"""

from .__version__ import __version__
from .diff import Diff, compare_many, diff_manifests, render_diff, render_manifest
from .manifest import Manifest
from .run import Run, track, tracked
from .store import RunStore
from .sweep import (
    ReplayPlan, SweepResult, export_bundle, grid, load_bundle, plan_replay, sweep,
)

__all__ = [
    "__version__",
    "track", "tracked", "Run",
    "Manifest", "RunStore",
    "diff_manifests", "render_diff", "render_manifest", "Diff", "compare_many",
    "sweep", "grid", "SweepResult",
    "plan_replay", "ReplayPlan",
    "export_bundle", "load_bundle",
    "adapters",
]

from . import adapters  # noqa: E402  (needs the names above)
