"""cpm (Computational Psychiatry Modelling) adapter.

The cpm paper names the problem this package exists for: models are implemented
differently across labs with unstated assumptions, and undetected bugs propagate
as new researchers build on existing implementations. cpm solved model
*specification*. It did not solve knowing which fit produced which number.

What a generic tracker misses here:

* **Bounds and priors are the model.** Two fits with identical code and data but
  a learning-rate bound of ``(0, 1)`` versus ``(0, 2)`` are different
  experiments. cpm keeps these in ``Parameters``, not in the call arguments.
* **The estimator is a result-changing choice.** Fmin, FminBound, genetic,
  BADS, and the hierarchical variants converge differently. Which one ran, with
  which scipy method and tolerances, has to be in the record.
* **Fits have convergence status per participant.** A group-level parameter
  computed from 60 fits of which 7 failed to converge is not the same number as
  one where all 60 converged, and nothing in the output distinguishes them.
* **Participant count and identifiers.** Dropping two subjects changes every
  group statistic. The count and a hash of the identifier list are recorded.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..run import Run
from .base import safe

name = "cpm"


def is_available() -> bool:
    try:
        import cpm  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def describe_parameters(parameters: Any, run: Run, prefix: str = "model") -> None:
    """Record every free parameter's value, bounds, and prior."""
    names = safe(lambda: list(parameters.keys()), [])
    run.log_param(f"{prefix}.parameter_names", sorted(names))
    run.log_param(f"{prefix}.n_parameters", len(names))

    free = safe(lambda: list(parameters.free()), None)
    if free is not None:
        run.log_param(f"{prefix}.n_free_parameters", len(free))

    # bounds() returns [[lowers], [uppers]] in cpm.
    bounds = safe(lambda: parameters.bounds())
    if bounds is not None:
        try:
            lowers, uppers = bounds[0], bounds[1]
            for i, nm in enumerate(safe(lambda: list(parameters.free()), names)):
                run.log_param(f"{prefix}.bounds.{nm}", f"[{lowers[i]}, {uppers[i]}]")
        except Exception:
            run.log_param(f"{prefix}.bounds", str(bounds))

    for nm in names:
        value = safe(lambda n=nm: getattr(parameters, n))
        if value is None:
            continue
        run.log_param(f"{prefix}.value.{nm}", safe(lambda v=value: float(v), str(value)))
        prior = safe(lambda v=value: getattr(v, "prior", None))
        if prior is not None:
            run.log_param(
                f"{prefix}.prior.{nm}",
                safe(lambda p=prior: getattr(p, "dist", type(p).__name__), str(prior)),
            )
            args = safe(lambda v=value: getattr(v, "args", None))
            if args:
                run.log_param(f"{prefix}.prior_args.{nm}", args)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def describe_data(data: Any, run: Run, prefix: str = "data") -> None:
    """Record shape and a stable hash of participant identifiers.

    Hashing the identifier list rather than storing it keeps subject IDs out of
    the manifest -- these are behavioural studies and manifests get committed to
    public repositories -- while still detecting a changed or reordered cohort.
    """
    run.log_param(f"{prefix}.type", type(data).__name__)

    n = safe(lambda: len(data))
    if n is not None:
        run.log_param(f"{prefix}.n_records", n)

    columns = safe(lambda: sorted(map(str, data.columns)))
    if columns:
        run.log_param(f"{prefix}.columns", columns)

    for id_col in ("ppt", "participant", "subject", "id"):
        ids = safe(lambda c=id_col: sorted(map(str, data[c].unique())))
        if ids:
            run.log_param(f"{prefix}.n_participants", len(ids))
            digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]
            run.log_param(f"{prefix}.participant_id_sha256", digest)
            break

    if safe(lambda: "observed" in data) or safe(lambda: "observed" in data.columns):
        run.log_param(f"{prefix}.has_observed", True)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def describe_optimiser(optimiser: Any, run: Run, prefix: str = "fit") -> None:
    """Record which estimator ran and how it was configured."""
    run.log_param(f"{prefix}.estimator", type(optimiser).__name__)
    run.log_param(
        f"{prefix}.loss",
        safe(lambda: getattr(optimiser.loss, "__name__", str(optimiser.loss)), "unknown"),
    )
    run.log_param(f"{prefix}.uses_prior", bool(safe(lambda: optimiser.prior, False)))

    # These are constructor arguments stored as plain attributes, not in
    # ``kwargs``, so a generic tracker that only reads kwargs misses them.
    #
    # ``number_of_starts`` is the important one. With more than one start and
    # no explicit initial_guess, cpm draws initial guesses at random -- so the
    # same data, same bounds and same estimator can converge to different
    # optima across runs. Recording the restart count and whether the guess was
    # supplied or drawn is the difference between "this fit is irreproducible"
    # and "this fit is irreproducible *and here is why*".
    run.log_param(
        f"{prefix}.number_of_starts",
        safe(lambda: optimiser.__number_of_starts__, None)
        or safe(lambda: getattr(optimiser, "number_of_starts", 1), 1),
    )
    guess = safe(lambda: optimiser.initial_guess)
    run.log_param(f"{prefix}.initial_guess_supplied", guess is not None)
    if guess is not None:
        run.log_param(f"{prefix}.initial_guess", guess)

    run.log_param(f"{prefix}.parallel", bool(safe(lambda: optimiser.__parallel__, False)))
    cores = safe(lambda: optimiser.cl)
    if cores is not None:
        run.log_param(f"{prefix}.cores", cores)
    ppt = safe(lambda: optimiser.ppt_identifier)
    if ppt is not None:
        run.log_param(f"{prefix}.ppt_identifier", ppt)
    run.log_param(
        f"{prefix}.libraries", safe(lambda: sorted(optimiser.__libraries__), [])
    )

    kwargs = safe(lambda: dict(optimiser.kwargs or {}), {})
    for k, v in sorted(kwargs.items()):
        run.log_param(f"{prefix}.kwargs.{k}", v)

    # The scipy method and tolerance hide inside kwargs and change the answer.
    for key in ("method", "tol", "maxiter", "options"):
        if key in kwargs:
            run.log_param(f"{prefix}.{key}", kwargs[key])

    model = safe(lambda: optimiser.model)
    if model is not None:
        params = safe(lambda: model.parameters)
        if params is not None:
            describe_parameters(params, run, prefix="model")
        data = safe(lambda: model.data)
        if data is not None:
            describe_data(data, run, prefix="data")


def describe_fit_results(optimiser: Any, run: Run, prefix: str = "fit") -> None:
    """Record convergence and group-level outcomes after ``optimise()``.

    Convergence counts matter more than they look. A group mean over 60
    participants of whom 7 hit the iteration limit is a different number from
    one where all converged, and the output alone does not say which you have.
    """
    fits = safe(lambda: list(optimiser.fit), [])
    run.log_result(f"{prefix}.n_fits", len(fits))
    if not fits:
        return

    converged = 0
    unknown = 0
    for f in fits:
        status = None
        for key in ("success", "converged", "status"):
            if isinstance(f, dict) and key in f:
                status = f[key]
                break
        if status is None:
            unknown += 1
        elif bool(status) is True or status == 0:
            converged += 1
    run.log_result(f"{prefix}.n_converged", converged)
    if unknown:
        run.log_result(f"{prefix}.n_convergence_unknown", unknown)

    # Group-level central tendency of each fitted parameter.
    def _summaries():
        import numpy as np
        params = safe(lambda: list(optimiser.parameters), [])
        if not params:
            return {}
        keys = set()
        for p in params:
            if isinstance(p, dict):
                keys |= set(p)
        out = {}
        for k in sorted(keys):
            vals = [p[k] for p in params if isinstance(p, dict) and k in p]
            numeric = [float(v) for v in vals if isinstance(v, (int, float))]
            if numeric:
                out[f"group_mean.{k}"] = round(float(np.mean(numeric)), 6)
                out[f"group_sd.{k}"] = round(float(np.std(numeric)), 6)
        return out

    for k, v in safe(_summaries, {}).items():
        run.log_result(f"{prefix}.{k}", v)

    def _loss():
        import numpy as np
        vals = []
        for f in fits:
            if isinstance(f, dict):
                for key in ("fun", "loss", "nll", "value"):
                    if key in f and isinstance(f[key], (int, float)):
                        vals.append(float(f[key]))
                        break
        return round(float(np.mean(vals)), 6) if vals else None

    mean_loss = safe(_loss)
    if mean_loss is not None:
        run.log_result(f"{prefix}.mean_loss", mean_loss)


def optimise(optimiser: Any, run: Run, **kwargs: Any):
    """Run ``optimiser.optimise()`` with configuration and outcome recorded.

    ::

        with daftar.track("bandit-fit", seed=7) as run:
            cpm_adapter.optimise(fmin, run)
    """
    import cpm

    run.log_param("fit.cpm_version", safe(lambda: cpm.__version__, "unknown"))
    describe_optimiser(optimiser, run)
    result = optimiser.optimise(**kwargs)
    describe_fit_results(optimiser, run)
    return result


def describe(obj: Any, run: Run, prefix: str = "model") -> None:
    """Dispatch on whatever cpm object is handed in."""
    cls = type(obj).__name__
    if cls in ("Parameters",):
        describe_parameters(obj, run, prefix)
    elif cls in ("Wrapper", "Simulator"):
        params = safe(lambda: obj.parameters)
        if params is not None:
            describe_parameters(params, run, prefix)
        data = safe(lambda: obj.data)
        if data is not None:
            describe_data(data, run)
        run.log_param(f"{prefix}.generator", cls)
    else:
        describe_optimiser(obj, run)
