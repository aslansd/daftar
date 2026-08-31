"""How each adapter is meant to be used.

These are reference snippets, not runnable here -- each needs its framework
installed. Run this file to see which adapters are live in your environment.
"""

# ---------------------------------------------------------------- Jaxley ---
JAXLEY = '''
import jaxley as jx
from jaxley.channels import HH
import daftar
from daftar.adapters import jaxley as jxa

comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=4)
cell = jx.Cell(branch, parents=[-1, 0, 0])
cell.insert(HH())
cell.branch(0).loc(0.0).record()
cell.branch(0).loc(0.0).stimulate(jx.step_current(1.0, 2.0, 0.1, 0.025, 10.0))

with daftar.track("hh-cell", seed=0) as run:
    v = jxa.integrate(cell, run, t_max=10.0, delta_t=0.025)

# Recorded automatically, none of which is visible in the call arguments:
#   morphology.n_compartments, .n_branches, .channels, .n_synapses
#   integrate.solver = bwd_euler        integrate.solver.was_default = true
#   integrate.voltage_solver = jaxley.dhs
#   env.jax_enable_x64, env.jax_platform
#   result.voltage.v_mean / v_min / v_max / n_nonfinite
#
# The `.was_default` flags matter: if Jaxley changes a default in 0.7, every
# result moves and the diff shows exactly which field caused it.
'''

# ------------------------------------------------------------------- cpm ---
CPM = '''
from cpm.generators import Parameters, Wrapper
from cpm.optimisation import Fmin, minimise
import daftar
from daftar.adapters import cpm as cpma

parameters = Parameters(
    alpha=Value(value=0.5, lower=0.0, upper=1.0, prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25}),
    temperature=Value(value=1.0, lower=0.0, upper=10.0),
)
model = Wrapper(model=my_trial_fn, data=participant_data, parameters=parameters)
fmin = Fmin(model=model, data=all_data, minimisation=minimise.LogLikelihood.bernoulli)

with daftar.track("bandit-fit", seed=7) as run:
    run.add_input("data/bandit_trials.csv")
    cpma.optimise(fmin, run)

# Recorded automatically:
#   model.bounds.alpha = [0.0, 1.0]     model.prior.alpha, model.prior_args.alpha
#   fit.estimator = Fmin                fit.loss = bernoulli
#   fit.method, fit.tol, fit.maxiter    (pulled out of kwargs)
#   data.n_participants, data.participant_id_sha256
#   result.fit.n_converged / n_fits     result.fit.group_mean.alpha
#
# n_converged is the one to watch. A group mean over 60 participants of whom
# 7 hit the iteration limit is a different number from one where all converged,
# and nothing in the output tells you which you have.
'''

# ------------------------------------------------------------ MeltingPot ---
MELTINGPOT = '''
import daftar
from daftar.adapters import meltingpot as mpa

roles = ["default"] * 5
with daftar.track("commons-harvest", seed=1234) as run:
    substrate = mpa.build("commons_harvest__open", roles, run)
    mpa.describe_scenario("commons_harvest__open_0", run)
    stats = mpa.run_episode(substrate, my_policy, run, max_steps=1000)

# Recorded automatically:
#   substrate.config_sha256    (detects a mutated ConfigDict)
#   substrate.roles, .n_players, .max_episode_frames, .n_actions
#   scenario.bots, .bots_sha256, .n_focal, .n_bots
#   result.episode.total_return, .mean_return, .gini
#   result.episode.return.player_0 ... player_4
#   result.episode.truncated   (hit max_steps without terminating)
#
# Per-player returns, not just the total: a substrate where everyone scores 10
# and one where a single player takes 70 have the same sum and are opposite
# findings. Gini is recorded for the same reason.
'''

# ------------------------------------------------------------- Concordia ---
CONCORDIA_NOTE = '''
Concordia is deliberately not an adapter in v0.1.

Every agent step calls LanguageModel.sample_text(). No major provider
guarantees token-level determinism even with a fixed seed, so `replay` cannot
mean what it means everywhere else in this package. Shipping an adapter whose
replay silently does not replay would undermine the one property the tool is
selling.

The right design, for a later release, is a different contract: wrap the
LanguageModel, record a hash of every (prompt, response) pair in order, and
have `diff` report transcript divergence -- the first step at which two runs
took different paths -- rather than claiming reproducibility. That turns
Concordia from an awkward fit into the strongest demonstration of why the
provenance layer matters at all: LLM-driven simulation is the case where nobody
can currently audit anything.

Build it after the deterministic adapters have users.
'''


def main():
    from daftar import adapters

    print("Adapter status in this environment\n")
    for name in adapters.registry.all():
        mod = adapters.get(name)
        mark = "available" if mod.is_available() else "not installed"
        print(f"  {name:<12} {mark}")

    print("\nReference usage:")
    for title, snippet in (
        ("Jaxley", JAXLEY), ("cpm", CPM), ("MeltingPot", MELTINGPOT),
    ):
        print(f"\n{'-' * 70}\n{title}\n{'-' * 70}{snippet}")
    print(f"\n{'-' * 70}\nConcordia -- why it is not here yet\n{'-' * 70}{CONCORDIA_NOTE}")


if __name__ == "__main__":
    main()
