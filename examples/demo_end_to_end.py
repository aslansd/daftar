"""End-to-end demo. No frameworks required -- runs anywhere.

Three scenarios, each one a question researchers actually ask:

1. I changed nothing. Did I get the same answer?
2. I changed one thing. Did it matter, and can I prove that's what did it?
3. I changed nothing and got a different answer. What now?
"""

import random
import shutil
import sys
from pathlib import Path

import daftar
from daftar.diff import diff_manifests, render_diff
from daftar.store import RunStore

HERE = Path(__file__).parent
STORE_DIR = HERE / ".demo-daftar"


def banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def leaky_integrate(dt=0.025, tau=10.0, noise=0.0, steps=2000):
    """A stand-in for a real simulation. Deterministic unless noise > 0."""
    v, total = 0.0, 0.0
    for i in range(steps):
        drive = 1.0 + (random.gauss(0, noise) if noise else 0.0)
        v += dt * (-v / tau + drive)
        total += v
    return {"v_final": round(v, 6), "v_mean": round(total / steps, 6)}


def main():
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    store = RunStore(STORE_DIR)
    store.init()

    data = HERE / "connectome.csv"
    data.write_text("pre,post,weight\n1,2,0.4\n2,3,0.7\n")

    # -- 1. baseline, then an honest repeat ------------------------------
    banner("1.  Same code, same seed, same inputs -- did it reproduce?")

    ids = []
    for _ in range(2):
        with daftar.track("leaky", params={"dt": 0.025, "tau": 10.0}, seed=42,
                      store=store) as run:
            run.add_input(data)
            run.log_results(leaky_integrate(dt=0.025, tau=10.0))
            ids.append(run.run_id)

    d = diff_manifests(store.load(ids[0]), store.load(ids[1]))
    print(render_diff(d))

    # -- 2. one deliberate change ----------------------------------------
    banner("2.  One parameter changed on purpose -- did it move the result?")

    with daftar.track("leaky", params={"dt": 0.010, "tau": 10.0}, seed=42,
                  store=store) as run:
        run.add_input(data)
        run.log_results(leaky_integrate(dt=0.010, tau=10.0))
        changed_id = run.run_id

    print(render_diff(diff_manifests(store.load(ids[0]), store.load(changed_id))))

    # -- 3. the case the tool exists for ---------------------------------
    banner("3.  Nothing recorded changed, but the answer did.")
    print("    (An unseeded noise term -- the kind of bug that survives for years.)\n")

    noisy = []
    for _ in range(2):
        with daftar.track("leaky", params={"dt": 0.025, "tau": 10.0}, seed=42,
                      store=store) as run:
            run.add_input(data)
            random.seed()  # simulates a library reseeding itself from the clock
            run.log_results(leaky_integrate(dt=0.025, tau=10.0, noise=0.3))
            noisy.append(run.run_id)

    print(render_diff(diff_manifests(store.load(noisy[0]), store.load(noisy[1]))))

    # -- 4. sweep ---------------------------------------------------------
    banner("4.  A sweep -- each point is a first-class run.")

    result = daftar.sweep(
        leaky_integrate, label="tau-sweep", seed=42, store=store,
        dt=[0.025], tau=[5.0, 10.0, 20.0],
    )
    rows = result.table(store=store)
    cols = [c for c in rows[0] if c != "status"]
    print("  " + "  ".join(c.ljust(16) for c in cols))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, ""))[:16].ljust(16) for c in cols))

    # -- 5. replay + export ----------------------------------------------
    banner("5.  What would it take to reproduce run 1?")
    print(daftar.plan_replay(store.load(ids[0])).render())

    banner("6.  Hand it to a successor.")
    bundle = daftar.export_bundle(store.load(ids[0]), HERE / "run.zip", store=store)
    print(f"  Wrote {bundle.name} ({bundle.stat().st_size / 1024:.1f} KiB)")
    print("  Contains: README.md, manifest.json, fields.tsv, inputs/connectome.csv")
    print("  None of which need daftar installed to read.")

    data.unlink(missing_ok=True)
    print()


if __name__ == "__main__":
    sys.exit(main())
