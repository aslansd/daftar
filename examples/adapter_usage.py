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

# ---------------------------------------------------------------- Brian2 ---
BRIAN2 = '''
from brian2 import *
import daftar
from daftar.adapters import brian2 as b2a

eqs = """dv/dt = (I-v)/tau : 1
I : 1
tau : second"""
G = NeuronGroup(100, eqs, threshold="v>1", reset="v=0", name="G")
G.I = "1.5 + 0.5*rand()"; G.tau = 10*ms
S = Synapses(G, G, on_pre="v_post += 0.05", name="S"); S.connect(p=0.05)
net = Network(G, S, SpikeMonitor(G, name="spikes"))

with daftar.track("balanced-net", seed=42) as run:
    b2a.run_network(net, 1*second, run)

# Recorded automatically:
#   group.G.method_choice   = (exact, euler, heun)   <- what was permitted
#   group.G.method_resolved = exact                  <- what actually ran
#   brian2.codegen_resolved = cython (auto)          <- auto means machine-dependent
#   brian2.prefs.core.default_float_dtype = float64
#   network.schedule        = [start, groups, thresholds, synapses, resets, end]
#   synapses.S.n_synapses   = 496                    <- p=0.05 is drawn, not declared
#   result.monitor.spikes.num_spikes
#
# method_resolved is the one to watch. Brian2's default is a candidate list, not
# a method: a model that integrated `exact` silently falls back to `euler` after
# an edit that makes the equations non-linear. Brian2 stores the winner nowhere
# -- it only mentions it in a log line, once per process.
#
# Note also that daftar.track(seed=...) now calls brian2.seed(). Seeding numpy
# does not reach Brian2's device RNG, so without it `connect(p=0.05)` draws a
# different graph every run.
'''

# ------------------------------------------------------------------- MNE ---
MNE = '''
import mne
import daftar
from daftar.adapters import mne as mnea

raw = mne.io.read_raw_fif("sub-01_raw.fif", preload=True)
raw.info["bads"] = ["EEG 053"]

with daftar.track("preproc", seed=42) as run:
    mnea.describe_environment(run)
    mnea.filter_raw(raw, run, l_freq=1.0, h_freq=40.0)

    ica = mne.preprocessing.ICA(n_components=20, random_state=97)
    ica.fit(raw)
    ica.exclude = [0, 3]              # <- the decision nothing else records
    mnea.apply_ica(ica, raw, run)

    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
                        reject=dict(eeg=150e-6), preload=True)
    mnea.describe_epochs(epochs, run)
    mnea.describe_evoked(epochs.average(), run)

# Recorded automatically:
#   ica.exclude = [0, 3]            <- the human decision, otherwise unrecorded
#   ica.n_excluded = 2
#   ica.random_state = 97           <- None would read "none (NOT REPRODUCIBLE)"
#   result.ica.converged            <- n_iter_ == max_iter means it stopped, not finished
#   filter.phase = zero             filter.phase.was_default = true
#   filter.fir_design, .l_trans_bandwidth, .h_trans_bandwidth
#   recording.bads, .n_bads         <- another human judgement
#   result.epochs.n_epochs_dropped, .drop_rate, .drop_reasons
#
# ica.exclude is the point. A reviewer asking "which components did you remove?"
# is usually asking a question with no surviving answer. raw.filter() is the
# same shape of problem: info["highpass"]/["lowpass"] survive, the design that
# produced them does not.
#
# On privacy: subject_info and file paths are hashed, never stored, and
# meas_date is recorded only as present/absent. Dates of service are
# identifiers, and manifests get committed to public repositories.
'''

# ------------------------------------------------------------------- sbi ---
SBI = '''
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
import daftar
from daftar.adapters import sbi as sbia

prior = BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2))
theta = prior.sample((1000,)); x = simulator(theta)

with daftar.track("npe", seed=42) as run:
    inference = NPE(prior=prior, density_estimator="maf")
    sbia.append_simulations(inference, theta, x, run)
    estimator = sbia.train(inference, run, training_batch_size=200,
                           learning_rate=5e-4, max_num_epochs=500)
    posterior = inference.build_posterior(estimator)
    samples = sbia.sample_posterior(posterior, (10_000,), run, x=x_o)

# Recorded automatically:
#   training.learning_rate, .training_batch_size, .stop_after_epochs, ...
#   training.stop_after_epochs.was_default = true
#   result.training.converged        <- False means it hit max_num_epochs
#   result.training.best_validation_loss
#   estimator.class, .n_parameters, .input_shape, .embedding_net
#   sbi.round_0.proposal = prior     sbi.round_1.proposal = DirectPosterior
#   sbi.num_simulations_per_round, .num_simulations_total
#   prior.type, .n_dims, .low, .high
#   posterior.x_o_sha256             <- hashed, not stored
#   result.posterior.mean / .std
#
# sbi keeps none of the train() arguments after the call, so two posteriors
# trained at different learning rates are otherwise indistinguishable. And
# `converged` matters: stopping because validation loss plateaued and stopping
# because you ran out of epochs are different outcomes, and sbi only warns.
#
# In sequential methods the proposal IS the algorithm -- round 1 draws from the
# prior, later rounds from the posterior. sbi records how many rounds happened
# but not what each drew from.
'''

# --------------------------------------------------------------- Nilearn ---
NILEARN = '''
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import daftar
from daftar.adapters import nilearn as nla

confounds = pd.read_csv("sub-01_desc-confounds_timeseries.tsv", sep="\t")
keep = [c for c in confounds if c.startswith(("trans_", "rot_"))] + ["csf"]

with daftar.track("connectivity", seed=42) as run:
    nla.describe_environment(run)
    nla.describe_image(func_img, run)

    masker = NiftiLabelsMasker(atlas, standardize="zscore_sample", t_r=2.0)
    ts = nla.fit_transform(masker, func_img, run, confounds=confounds[keep])

    measure = ConnectivityMeasure(kind="correlation", vectorize=True)
    conn = nla.connectivity_fit_transform(measure, [ts], run)

# Recorded automatically:
#   confounds.names, .n_regressors, .n_motion, .n_tissue, .n_compcor
#   confounds.values_sha256           <- values hashed, never stored
#   masker.mask_resolved, .mask_n_voxels   <- mask_img=None computes per subject
#   masker.n_labels, .labels_sha256   <- how many regions actually resolved
#   connectivity.kind = correlation
#   connectivity.cov_estimator = None
#   connectivity.cov_estimator_resolved = LedoitWolf   <- shrinkage, declared None
#   result.connectivity.mean, .abs_max
#
# The confounds are the point. fit_transform(confounds=...) regresses them out
# and stores nothing -- get_params() has no `confounds` key -- yet which columns
# you chose determines every connectivity value. It is the ica.exclude of fMRI.
#
# cov_estimator is the subtle one: declared None, resolved to Ledoit-Wolf
# shrinkage, which pulls the covariance toward the identity and on weakly
# correlated data can dominate the result entirely.
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
        ("Jaxley", JAXLEY), ("cpm", CPM), ("Brian2", BRIAN2), ("MNE-Python", MNE), ("sbi", SBI), ("Nilearn", NILEARN),
        ("MeltingPot", MELTINGPOT),
    ):
        print(f"\n{'-' * 70}\n{title}\n{'-' * 70}{snippet}")
    print(f"\n{'-' * 70}\nConcordia -- why it is not here yet\n{'-' * 70}{CONCORDIA_NOTE}")


if __name__ == "__main__":
    main()
