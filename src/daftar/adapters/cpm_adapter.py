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

def _describe_prior(prior: Any) -> tuple[str, Any]:
    """Name and parameters of a cpm prior.

    ``Value.prior`` is not the string you passed to the constructor -- cpm turns
    ``prior="truncated_normal"`` into a *frozen scipy distribution*. Stringifying
    that object yields ``<scipy...truncnorm_gen object at 0x7f...>``: a memory
    address, different on every run, which would make every diff report a change
    that did not happen.

    Frozen distributions expose the useful parts as ``.dist.name`` (``truncnorm``,
    ``norm``, ``beta``) plus ``.args`` and ``.kwds`` for the shape, location and
    scale actually used. Those are the numbers that define the prior, and they
    are stable.
    """
    name = safe(lambda: prior.dist.name)
    if not name:
        name = safe(lambda: prior.name) or type(prior).__name__

    params: dict[str, Any] = {}
    kwds = safe(lambda: dict(prior.kwds), {}) or {}
    for k, v in kwds.items():
        params[k] = safe(lambda vv=v: round(float(vv), 8), vv_fallback(v))
    args = safe(lambda: list(prior.args), []) or []
    for i, v in enumerate(args):
        params[f"arg{i}"] = safe(lambda vv=v: round(float(vv), 8), vv_fallback(v))
    return str(name), params


def vv_fallback(v: Any) -> str:
    return "<unavailable>" if v is None else str(type(v).__name__)


def _scalar(value: Any) -> Any:
    """A stable representation of a parameter value.

    cpm's ``Value`` defines ``__float__``, so scalars convert cleanly. But cpm
    also allows vector-valued parameters (``Value(value=[0.1, 0.2, 0.3])``), and
    for those ``float()`` raises and the object is not iterable either -- the
    numbers live on ``.value``, and ``Value`` also implements ``__array__``.
    Falling back to ``str(value)`` would emit a memory address, so both routes
    to the underlying numbers are tried before giving up.
    """
    as_float = safe(lambda: float(value))
    if as_float is not None:
        return round(as_float, 8)

    # Unwrap a Value-like container, then treat it as a sequence.
    inner = safe(lambda: value.value)
    for candidate in (inner, value):
        if candidate is None:
            continue
        as_list = safe(lambda c=candidate: [round(float(x), 8) for x in c])
        if as_list is not None:
            return as_list

    as_array = safe(lambda: [round(float(x), 8) for x in __import__("numpy").asarray(value).ravel()])
    if as_array is not None:
        return as_array
    return "<non-numeric>"


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
            run.log_param(f"{prefix}.bounds", _scalar(bounds))

    for nm in names:
        value = safe(lambda n=nm: getattr(parameters, n))
        if value is None:
            continue
        run.log_param(f"{prefix}.value.{nm}", _scalar(value))
        prior = safe(lambda v=value: getattr(v, "prior", None))
        if prior is not None:
            dist_name, dist_params = _describe_prior(prior)
            run.log_param(f"{prefix}.prior.{nm}", dist_name)
            if dist_params:
                run.log_param(f"{prefix}.prior_args.{nm}", dist_params)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def describe_data(
    data: Any, run: Run, prefix: str = "data", groups: Any = None
) -> None:
    """Record shape and a stable hash of participant identifiers.

    Hashing the identifier list rather than storing it keeps subject IDs out of
    the manifest -- these are behavioural studies and manifests get committed to
    public repositories -- while still detecting a changed or reordered cohort.

    ``data`` may be a DataFrame, a DataFrameGroupBy (what cpm holds after
    ``prepare_data``), or a list of per-participant dicts. Pass ``groups`` when
    the caller already knows the cohort keys; it is authoritative.
    """
    run.log_param(f"{prefix}.type", type(data).__name__)

    # Cohort identifiers, in order of reliability.
    ids: list[str] | None = None
    if groups:
        ids = [str(g) for g in groups]
    else:
        keys = safe(lambda: list(data.groups.keys()))
        if keys:
            ids = [str(k) for k in keys]

    # A DataFrameGroupBy exposes the underlying frame as ``.obj``; len() on the
    # groupby itself counts groups, not rows, which would silently mislabel the
    # number of trials as the number of participants.
    frame = safe(lambda: data.obj)
    if frame is None:
        frame = data

    n_records = safe(lambda: len(frame))
    if n_records is not None:
        run.log_param(f"{prefix}.n_records", n_records)

    columns = safe(lambda: sorted(map(str, frame.columns)))
    if columns:
        run.log_param(f"{prefix}.columns", columns)

    if ids is None:
        for id_col in ("ppt", "participant", "subject", "id"):
            found = safe(lambda c=id_col: sorted(map(str, frame[c].unique())))
            if found:
                ids = found
                break

    if ids is None and isinstance(data, list):
        # A list of per-participant records: the length is the cohort size.
        ids = [str(i) for i in range(len(data))]

    if ids is not None:
        run.log_param(f"{prefix}.n_participants", len(ids))
        digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:12]
        run.log_param(f"{prefix}.participant_id_sha256", digest)

    has_observed = safe(lambda: "observed" in frame.columns, False)
    if has_observed:
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
    # ``number_of_starts`` is never retained by cpm: it is consumed in
    # ``__init__`` to build ``initial_guess`` with shape
    # ``(number_of_starts, n_free_params)`` and then discarded. cpm itself
    # recovers it as ``self.initial_guess.shape[0]``, so we do the same.
    guess = safe(lambda: optimiser.initial_guess)
    if guess is not None:
        run.log_param(
            f"{prefix}.number_of_starts", safe(lambda: len(guess), "unknown")
        )
        # Record the guesses themselves rather than a "were these supplied?"
        # flag. cpm keeps no record of whether guesses were user-supplied or
        # drawn at random, so that flag is not recoverable -- but the values
        # are, and they are strictly more useful. If they were drawn randomly
        # they differ between runs, so a diff shows
        # ``param.fit.initial_guess`` as a candidate cause of a different fit
        # instead of leaving the divergence unexplained.
        run.log_param(
            f"{prefix}.initial_guess",
            safe(lambda: [[round(float(v), 8) for v in row] for row in guess],
                 "<unavailable>"),
        )
    else:
        run.log_param(f"{prefix}.number_of_starts", 1)

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

    # The cohort lives on the *optimiser*, not on the model.
    #
    # A cpm Wrapper holds one participant's trials as a template -- cpm calls
    # `model.reset(data=participant)` for each subject in turn. Reading
    # `optimiser.model.data` therefore reports a cohort of 1 no matter how many
    # people are in the study, which is precisely the kind of confidently wrong
    # provenance that is worse than recording nothing at all.
    #
    # `optimiser.groups` is the authoritative list of cohort keys. Note that
    # cpm's attribute named `participants` is *not* the participant list: it is
    # the first group's DataFrame, kept as a template.
    cohort = safe(lambda: optimiser.data)
    if cohort is not None:
        describe_data(
            cohort, run, prefix="data", groups=safe(lambda: optimiser.groups),
        )


def _is_converged(status: Any) -> bool:
    """Interpret a convergence flag across the conventions cpm may hand back.

    The subtlety that bit this once: in Python ``False == 0`` is ``True``. A
    naive ``status is True or status == 0`` therefore counts *failed* fits as
    converged, which inflates ``n_converged`` to the total every time. A group
    mean silently computed over non-converged fits is exactly the confidently
    wrong number this adapter exists to prevent, so booleans are tested before
    integers -- and note ``isinstance(True, int)`` is also ``True``, hence the
    order.
    """
    if isinstance(status, bool) or type(status).__name__ == "bool_":
        return bool(status)
    if isinstance(status, int):
        return status == 0          # scipy convention: 0 means success
    return bool(status)


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
        elif _is_converged(status):
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
