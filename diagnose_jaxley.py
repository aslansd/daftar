"""Print exactly why the Jaxley live tests are skipping.

    python diagnose_jaxley.py
"""
import importlib.metadata as md
import sys
import traceback

print("=" * 70)
for dist in ("jaxley", "jax", "jaxlib"):
    try:
        print(f"  {dist:<10} {md.version(dist)}")
    except Exception as exc:
        print(f"  {dist:<10} NOT INSTALLED ({exc})")

try:
    import jaxley as jx
    print(f"\n  jaxley loaded from: {jx.__file__}")
except Exception:
    print("\n  import jaxley FAILED:")
    traceback.print_exc()
    sys.exit(1)

# Is this the released 0.13.0 (broken) or a build with the fix?
try:
    import inspect

    from jaxley import solver_gate
    src = inspect.getsource(solver_gate.save_exp)
    print("\n  solver_gate.save_exp source:")
    for line in src.splitlines():
        print("    " + line)
    if "a_max" in src:
        print("\n  >>> This build still uses jnp.clip(a_max=...) -- the broken one.")
    else:
        print("\n  >>> This build is patched (no a_max). Good.")
except Exception:
    print("\n  could not read solver_gate.save_exp:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Running daftar's preflight:\n")
try:
    from tests.test_adapters_live import _jaxley_is_usable
except Exception:
    sys.path.insert(0, "tests")
    from test_adapters_live import _jaxley_is_usable

ok, why = _jaxley_is_usable()
print(f"  usable: {ok}")
print(f"  reason: {why or '(none)'}")

if not ok:
    print("\n  Full traceback from a direct attempt:\n")
    try:
        import jaxley as jx
        from jaxley.channels import HH

        cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
        cell.insert(HH())
        cell.branch(0).loc(0.0).record(verbose=False)
        jx.integrate(cell, t_max=0.1)
    except Exception:
        traceback.print_exc()
print("=" * 70)
