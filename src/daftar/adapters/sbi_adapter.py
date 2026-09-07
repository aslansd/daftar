"""sbi (simulation-based inference) adapter.

sbi has the same shape of problem as the other adapters here, in a more acute
form: **almost nothing that determines a posterior is stored on the objects
afterwards.**

* **Training hyperparameters vanish.** `train(training_batch_size=200,
  learning_rate=5e-4, stop_after_epochs=20, max_num_epochs=...)` configures the
  fit and is then discarded. Nothing on the trainer records what you passed, so
  two posteriors trained with different learning rates are indistinguishable.
  :func:`train` records them.

* **Whether training converged.** sbi stops either because validation loss
  stopped improving or because it hit `max_num_epochs` -- and those are
  completely different outcomes. It emits a `UserWarning` in the second case and
  stores nothing you would notice. `epochs_trained` sits in `summary`; the limit
  it was compared against does not.

* **The proposal is the method.** In multi-round SNPE/SNLE/SNRE, round 1 draws
  from the prior and later rounds draw from the current posterior. That
  sequence *is* the algorithm, and `_data_round_index` records only how many
  rounds happened, not what each drew from.

* **The density estimator is usually a string.** `density_estimator="maf"`
  becomes a flow with a particular depth, width and embedding net, chosen by
  defaults that change between sbi releases. The resolved architecture and its
  parameter count are what actually ran.

* **The prior is an experimental parameter.** Two inferences over different
  prior bounds are different experiments, and the bounds appear nowhere in a
  saved posterior's numbers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "sbi"


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("sbi")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _sha(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def _tensor_digest(tensor: Any) -> str | None:
    """Content hash of a tensor, so a changed observation is detectable.

    Hashing rather than storing: `x_o` can be large, and for some applications
    it is measured data that does not belong in a committed manifest.
    """
    def _hash():
        import numpy as np
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return safe(_hash)


#: ``train()`` defaults, recorded explicitly when not passed. sbi discards them,
#: so a future change of default would otherwise be invisible.
TRAIN_DEFAULTS = {
    "training_batch_size": 200,
    "learning_rate": 5e-4,
    "validation_fraction": 0.1,
    "stop_after_epochs": 20,
    "max_num_epochs": 2**31 - 1,
    "clip_max_norm": 5.0,
    "resume_training": False,
    "retrain_from_scratch": False,
    "discard_prior_samples": False,
}


# --------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------

def describe_prior(prior: Any, run: Run, prefix: str = "prior") -> None:
    """Record the prior: its family, dimensionality, and support.

    Two inferences over different prior bounds are different experiments. The
    bounds appear nowhere in a saved posterior's numbers.
    """
    run.log_param(f"{prefix}.type", type(prior).__name__)

    dim = safe(lambda: int(prior.event_shape[0]))
    if dim is None:
        dim = safe(lambda: int(prior.sample().numel()))
    run.log_param(f"{prefix}.n_dims", dim if dim is not None else "unknown")

    # BoxUniform and friends expose support bounds; record them exactly.
    for attr, key in (("low", "low"), ("high", "high")):
        value = safe(lambda a=attr: getattr(prior, a, None))
        if value is None:
            base = safe(lambda a=attr: getattr(prior.base_dist, a, None))
            value = base
        if value is not None:
            run.log_param(
                f"{prefix}.{key}",
                safe(lambda v=value: [round(float(x), 8) for x in v.flatten()],
                     "<unavailable>"),
            )

    mean = safe(lambda: prior.mean)
    if mean is not None:
        run.log_param(f"{prefix}.mean_sha256", _sha(safe(lambda: mean.tolist())))


# --------------------------------------------------------------------------
# density estimator
# --------------------------------------------------------------------------

def describe_estimator(estimator: Any, run: Run, prefix: str = "estimator") -> None:
    """Record the *resolved* architecture, not the string that requested it."""
    run.log_param(f"{prefix}.class", type(estimator).__name__)

    inner = safe(lambda: estimator.net)
    if inner is not None:
        run.log_param(f"{prefix}.net_class", type(inner).__name__)

    for attr in ("input_shape", "condition_shape"):
        value = safe(lambda a=attr: getattr(estimator, a, None))
        if value is not None:
            run.log_param(f"{prefix}.{attr}",
                          safe(lambda v=value: list(map(int, v)), str(value)))

    embedding = safe(lambda: estimator.embedding_net)
    if embedding is not None:
        run.log_param(f"{prefix}.embedding_net", type(embedding).__name__)
        run.log_param(
            f"{prefix}.embedding_n_params",
            safe(lambda: sum(p.numel() for p in embedding.parameters()), 0),
        )

    # Parameter count is a coarse but honest fingerprint of the architecture:
    # it changes when depth, width or the embedding changes, including when an
    # sbi default changes underneath you.
    run.log_param(
        f"{prefix}.n_parameters",
        safe(lambda: sum(p.numel() for p in estimator.parameters()), "unknown"),
    )


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------

def describe_trainer(inference: Any, run: Run, prefix: str = "sbi") -> None:
    """Record the inference method, device, and per-round simulation counts."""
    from importlib.metadata import version

    run.log_param(f"{prefix}.version", safe(lambda: version("sbi"), "unknown"))
    run.log_param(f"{prefix}.method", type(inference).__name__)
    run.log_param(f"{prefix}.device", str(safe(lambda: inference._device, "unknown")))

    round_index = safe(lambda: list(inference._data_round_index), [])
    if round_index:
        run.log_param(f"{prefix}.n_rounds", len(set(round_index)))
        run.log_param(f"{prefix}.round_index", round_index)

    # Simulation budget, per round and total. The budget is the single most
    # reported number in an sbi paper and is not stored anywhere retrievable.
    thetas = safe(lambda: list(inference._theta_roundwise), [])
    if thetas:
        per_round = safe(lambda: [int(t.shape[0]) for t in thetas], [])
        run.log_param(f"{prefix}.num_simulations_per_round", per_round)
        run.log_param(f"{prefix}.num_simulations_total", sum(per_round))

    prior = safe(lambda: inference._prior)
    if prior is not None:
        describe_prior(prior, run)


def append_simulations(inference: Any, theta: Any, x: Any, run: Run,
                       proposal: Any = None, prefix: str = "sbi", **kwargs: Any):
    """``append_simulations`` with the round and its proposal recorded.

    In multi-round methods the proposal *is* the algorithm: round 1 draws from
    the prior, later rounds from the current posterior. sbi records how many
    rounds happened but not what each one drew from, so an amortised run and a
    sequential run with the same simulation budget look alike afterwards.
    """
    round_no = len(set(safe(lambda: list(inference._data_round_index), []))) 
    run.log_param(f"{prefix}.round_{round_no}.n_simulations",
                  safe(lambda: int(theta.shape[0]), "unknown"))
    run.log_param(f"{prefix}.round_{round_no}.proposal",
                  "prior" if proposal is None else type(proposal).__name__)
    run.log_param(f"{prefix}.round_{round_no}.theta_sha256", _tensor_digest(theta))
    run.log_param(f"{prefix}.round_{round_no}.x_sha256", _tensor_digest(x))

    if proposal is None:
        return inference.append_simulations(theta, x, **kwargs)
    return inference.append_simulations(theta, x, proposal=proposal, **kwargs)


def train(inference: Any, run: Run, prefix: str = "training", **kwargs: Any):
    """``inference.train()`` with hyperparameters and convergence recorded.

    ::

        with daftar.track("npe", seed=42) as run:
            sbi_adapter.append_simulations(inference, theta, x, run)
            estimator = sbi_adapter.train(inference, run, max_num_epochs=200)

    sbi keeps none of these arguments after the call. Two posteriors trained
    with different learning rates or different early-stopping patience are
    otherwise indistinguishable.
    """
    resolved = dict(TRAIN_DEFAULTS)
    resolved.update(kwargs)

    for key, value in sorted(resolved.items()):
        run.log_param(f"{prefix}.{key}", value)
        if key in TRAIN_DEFAULTS and key not in kwargs:
            run.log_param(f"{prefix}.{key}.was_default", True)

    describe_trainer(inference, run)
    estimator = inference.train(**kwargs)
    describe_estimator(estimator, run)
    describe_training_outcome(inference, run, resolved.get("max_num_epochs"), prefix)
    return estimator


def describe_training_outcome(inference: Any, run: Run,
                              max_num_epochs: Any = None,
                              prefix: str = "training") -> None:
    """Record how training ended, which sbi only ever warns about.

    Stopping because validation loss plateaued and stopping because the epoch
    limit was reached are completely different outcomes. sbi raises a
    ``UserWarning`` for the second and stores nothing you would notice
    afterwards, so a posterior from a truncated fit looks like any other.
    """
    summary = safe(lambda: dict(inference.summary), {}) or {}

    epochs = safe(lambda: [int(e) for e in summary.get("epochs_trained", [])], [])
    if epochs:
        run.log_result(f"{prefix}.epochs_trained", epochs)
        run.log_result(f"{prefix}.epochs_trained_last", epochs[-1])
        if isinstance(max_num_epochs, int):
            run.log_result(f"{prefix}.converged", epochs[-1] < max_num_epochs)

    losses = safe(
        lambda: [round(float(v), 6) for v in summary.get("best_validation_loss", [])],
        [],
    )
    if losses:
        run.log_result(f"{prefix}.best_validation_loss", losses)
        run.log_result(f"{prefix}.best_validation_loss_last", losses[-1])

    durations = safe(
        lambda: [round(float(v), 3) for v in summary.get("epoch_durations_sec", [])],
        [],
    )
    if durations:
        run.log_result(f"{prefix}.total_train_seconds", round(sum(durations), 3))


# --------------------------------------------------------------------------
# posterior
# --------------------------------------------------------------------------

def describe_posterior(posterior: Any, run: Run, x_o: Any = None,
                       prefix: str = "posterior") -> None:
    """Record the posterior object and the observation it is conditioned on.

    ``x_o`` is hashed rather than stored: it can be large, and in applied work
    it is measured data that does not belong in a committed manifest. The hash
    still detects a swapped or edited observation, which is the failure this is
    guarding against.
    """
    run.log_param(f"{prefix}.class", type(posterior).__name__)

    sampler = safe(lambda: posterior._sample_with) or safe(
        lambda: type(posterior.posterior_estimator).__name__
    )
    if sampler:
        run.log_param(f"{prefix}.sampler", str(sampler))

    observation = x_o if x_o is not None else safe(lambda: posterior.default_x)
    if observation is not None:
        run.log_param(f"{prefix}.x_o_sha256", _tensor_digest(observation))
        run.log_param(f"{prefix}.x_o_shape",
                      safe(lambda: list(observation.shape), "unknown"))
    else:
        run.log_param(f"{prefix}.x_o_set", False)


def sample_posterior(posterior: Any, shape: tuple, run: Run, x: Any = None,
                     prefix: str = "posterior", **kwargs: Any):
    """``posterior.sample()`` with the observation and summary statistics recorded."""
    describe_posterior(posterior, run, x_o=x, prefix=prefix)
    run.log_param(f"{prefix}.n_samples_requested", int(shape[0]))

    samples = (posterior.sample(shape, x=x, **kwargs) if x is not None
               else posterior.sample(shape, **kwargs))

    def _stats():
        import numpy as np
        arr = np.asarray(samples.detach().cpu().numpy())
        return {
            "mean": [round(float(v), 6) for v in arr.mean(axis=0).flatten()],
            "std": [round(float(v), 6) for v in arr.std(axis=0).flatten()],
            "n_nonfinite": int((~np.isfinite(arr)).sum()),
        }

    for key, value in (safe(_stats, {}) or {}).items():
        run.log_result(f"{prefix}.{key}", value)
    return samples


def describe_environment(run: Run, prefix: str = "sbi") -> None:
    from importlib.metadata import version

    run.log_param(f"{prefix}.version", safe(lambda: version("sbi"), "unknown"))
    run.log_param(f"{prefix}.torch_version", safe(lambda: version("torch"), "unknown"))


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever sbi object is handed in."""
    cls = type(obj).__name__
    if "Posterior" in cls:
        describe_posterior(obj, run, prefix=prefix or "posterior")
    elif cls in ("NPE", "NLE", "NRE", "SNPE_A", "SNPE_C", "NPE_C", "NLE_A", "NRE_A",
                 "NRE_B", "NRE_C", "FMPE", "NPSE"):
        describe_trainer(obj, run, prefix or "sbi")
    elif hasattr(obj, "parameters"):
        describe_estimator(obj, run, prefix or "estimator")
    else:
        describe_prior(obj, run, prefix or "prior")
