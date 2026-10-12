"""Nilearn adapter.

A note on why this exists, since an earlier version of this project's roadmap
said it should not. That entry read "Nilearn / fMRIPrep -- fMRIPrep already
emits good BIDS-derivative provenance", and it conflated two different tools.
fMRIPrep is a *preprocessing* pipeline and its provenance is genuinely good --
but it stops exactly where nilearn begins. Everything that happens afterwards --
which confounds were regressed out, which atlas, which masker settings, which
GLM design -- is as unrecorded as anything else in this package.

Nilearn objects are scikit-learn estimators, so ``get_params()`` already exposes
the *declared* configuration and a generic tracker could capture that. What it
cannot reach:

* **The confounds.** ``masker.fit_transform(img, confounds=df)`` takes them as a
  *runtime argument*. They are regressed out and then forgotten -- nothing is
  stored on the masker, and ``get_params()`` has no ``confounds`` key. Which
  columns you chose determines every connectivity value and every GLM
  coefficient downstream, and it is typically a hand-written list comprehension
  over an fMRIPrep TSV. This is the ``ica.exclude`` of fMRI.

* **The resolved mask.** With ``mask_img=None`` the mask is *computed from the
  data*, so it differs per subject and per ``mask_strategy``. The declared
  parameter says ``None``; the mask that ran has a specific voxel count.

* **`standardize` semantics.** nilearn changed the meaning of
  ``standardize=True`` and added ``"zscore_sample"`` and ``"psc"``. The recorded
  value plus the nilearn version is what identifies what was actually done.

* **Atlas identity.** How many regions a parcellation resolved to, which is not
  the same as the name of the atlas you asked for.

**On subject data.** fMRI recordings are health data, so this adapter records
structure and never content: file paths are hashed, confound *column names* are
recorded but never their values, and image data is summarised. Manifests get
committed to public repositories.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "nilearn"


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("nilearn")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _sha(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def _round(value: Any, digits: int = 6) -> Any:
    return safe(lambda: round(float(value), digits), value)


# --------------------------------------------------------------------------
# estimator parameters
# --------------------------------------------------------------------------

#: Constructor parameters that are bookkeeping rather than science. Recording
#: them would add noise to every diff without ever explaining a result.
_IGNORED_PARAMS = frozenset({
    "memory", "memory_level", "verbose", "reports", "n_jobs",
    "minimize_memory", "dtype",
})


def describe_estimator(estimator: Any, run: Run, prefix: str) -> None:
    """Record a nilearn estimator's declared parameters via ``get_params()``.

    This is the part a generic sklearn-aware tracker could also do. It is here
    so the manifest is complete, not because it is the interesting bit.
    """
    run.log_param(f"{prefix}.class", type(estimator).__name__)

    params = safe(lambda: estimator.get_params(), {}) or {}
    for key, value in sorted(params.items()):
        if key in _IGNORED_PARAMS:
            continue
        # Nested estimators and images stringify to something unstable; record
        # their type rather than a repr.
        if hasattr(value, "get_params") or hasattr(value, "get_fdata"):
            run.log_param(f"{prefix}.{key}", type(value).__name__)
        else:
            run.log_param(f"{prefix}.{key}", value)


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def describe_image(img: Any, run: Run, prefix: str = "image") -> None:
    """Record image geometry, and hash any path rather than storing it."""
    run.log_param(f"{prefix}.shape",
                  safe(lambda: list(map(int, img.shape)), "unknown"))
    run.log_param(f"{prefix}.n_volumes",
                  safe(lambda: int(img.shape[3]) if len(img.shape) > 3 else 1, 1))

    # The affine defines voxel-to-world mapping. A changed affine means the data
    # was resampled or reoriented, which moves every downstream number.
    run.log_param(
        f"{prefix}.affine_sha256",
        safe(lambda: _sha([round(float(v), 6)
                           for v in img.affine.flatten()]), "<unavailable>"),
    )
    run.log_param(
        f"{prefix}.voxel_size_mm",
        safe(lambda: [round(float(z), 4) for z in img.header.get_zooms()[:3]],
             "unknown"),
    )

    # Paths routinely contain subject identifiers.
    path = safe(lambda: str(img.get_filename()))
    if path and path != "None":
        run.log_param(f"{prefix}.source_sha256", _sha(path))


# --------------------------------------------------------------------------
# confounds -- the reason this adapter exists
# --------------------------------------------------------------------------

def describe_confounds(confounds: Any, run: Run, prefix: str = "confounds") -> None:
    """Record which confounds were regressed out. Names only, never values.

    ``fit_transform(img, confounds=df)`` takes these as a runtime argument;
    nothing is stored on the masker afterwards. Which columns you chose is one
    of the largest free parameters in fMRI analysis -- 6 motion regressors, 24
    with derivatives and squares, aCompCor, global signal -- and each choice
    gives materially different connectivity.

    Column *names* are recorded because they are the decision. Column *values*
    are subject data and are hashed, never stored.
    """
    if confounds is None:
        run.log_param(f"{prefix}.applied", False)
        return

    run.log_param(f"{prefix}.applied", True)

    names = safe(lambda: [str(c) for c in confounds.columns])
    if names is None:
        # A plain array: no names available, so record the shape.
        run.log_param(f"{prefix}.type", type(confounds).__name__)
        run.log_param(f"{prefix}.n_regressors",
                      safe(lambda: int(confounds.shape[1]), "unknown"))
        return

    run.log_param(f"{prefix}.n_regressors", len(names))
    run.log_param(f"{prefix}.names", sorted(names))
    run.log_param(f"{prefix}.names_sha256", _sha(sorted(names)))

    # A coarse taxonomy, so a diff can say "motion regressors changed" rather
    # than making the reader compare two 24-element lists.
    families = {
        "motion": ("trans_", "rot_"),
        "compcor": ("a_comp_cor", "t_comp_cor", "w_comp_cor", "c_comp_cor"),
        "tissue": ("csf", "white_matter", "global_signal"),
        "scrub": ("motion_outlier", "framewise", "std_dvars", "dvars"),
        "cosine": ("cosine",),
    }
    for family, prefixes in families.items():
        count = sum(1 for n in names
                    if any(n.lower().startswith(p) for p in prefixes))
        if count:
            run.log_param(f"{prefix}.n_{family}", count)

    # Values are subject data. Hash so a changed confound file is detectable
    # without the numbers entering the manifest.
    def _digest():
        import numpy as np
        arr = np.ascontiguousarray(np.asarray(confounds, dtype="float64"))
        arr = np.nan_to_num(arr, nan=0.0)
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    digest = safe(_digest)
    if digest:
        run.log_param(f"{prefix}.values_sha256", digest)


# --------------------------------------------------------------------------
# maskers
# --------------------------------------------------------------------------

def describe_masker(masker: Any, run: Run, prefix: str = "masker") -> None:
    """Record a masker's parameters and, after fitting, the mask that resolved.

    With ``mask_img=None`` the mask is computed from the data, so it differs
    between subjects and between ``mask_strategy`` values. The declared
    parameter says ``None``; only the fitted attribute says what ran.
    """
    describe_estimator(masker, run, prefix)

    n_elements = safe(lambda: int(masker.n_elements_))
    if n_elements is not None:
        run.log_param(f"{prefix}.n_elements", n_elements)

    mask_img = safe(lambda: masker.mask_img_)
    if mask_img is not None:
        run.log_param(f"{prefix}.mask_resolved", True)
        run.log_param(
            f"{prefix}.mask_n_voxels",
            safe(lambda: int(__import__("numpy").asarray(
                mask_img.dataobj).astype(bool).sum()), "unknown"),
        )
        run.log_param(
            f"{prefix}.mask_shape",
            safe(lambda: list(map(int, mask_img.shape)), "unknown"),
        )

    # Parcellation identity: how many regions actually resolved, which is not
    # the same as the name of the atlas requested.
    labels = safe(lambda: list(masker.labels_))
    if labels is not None:
        run.log_param(f"{prefix}.n_labels", len(labels))
        run.log_param(f"{prefix}.labels_sha256", _sha(labels))

    region_names = safe(lambda: dict(masker.region_names_))
    if region_names:
        run.log_param(f"{prefix}.n_regions", len(region_names))


def fit_transform(masker: Any, imgs: Any, run: Run, confounds: Any = None,
                  prefix: str = "masker", **kwargs: Any):
    """``masker.fit_transform()`` with the confounds recorded.

    ::

        with daftar.track("connectivity", seed=42) as run:
            ts = nilearn_adapter.fit_transform(masker, img, run, confounds=conf)

    The confounds are the point. They are a runtime argument that nilearn
    regresses out and then forgets, and the choice of columns is one of the
    largest free parameters in the whole analysis.
    """
    describe_confounds(confounds, run)

    out = (masker.fit_transform(imgs, confounds=confounds, **kwargs)
           if confounds is not None else masker.fit_transform(imgs, **kwargs))

    describe_masker(masker, run, prefix)

    def _stats():
        import numpy as np
        arr = np.asarray(out)
        return {
            "shape": list(map(int, arr.shape)),
            "n_timepoints": int(arr.shape[0]),
            "n_signals": int(arr.shape[1]) if arr.ndim > 1 else 1,
            "n_nonfinite": int((~np.isfinite(arr)).sum()),
        }

    for key, value in (safe(_stats, {}) or {}).items():
        run.log_result(f"{prefix}.timeseries.{key}", value)
    return out


# --------------------------------------------------------------------------
# connectivity
# --------------------------------------------------------------------------

def describe_connectivity(measure: Any, run: Run, prefix: str = "connectivity") -> None:
    """Record the connectivity estimator, declared *and* resolved.

    ``kind`` changes the numbers entirely -- correlation, partial correlation
    and tangent are different quantities, not different formattings.

    The subtler one is ``cov_estimator``. It defaults to ``None`` and
    ``get_params()`` faithfully reports ``None`` -- but after fitting it has
    resolved to **Ledoit-Wolf shrinkage**, which pulls the covariance toward the
    identity. On weakly correlated data that shrinkage dominates: two analyses
    can differ in a preprocessing choice and still produce identical
    connectivity because both were shrunk to the same place. The declared
    parameter cannot tell you that happened; the fitted attribute can.
    """
    describe_estimator(measure, run, prefix)

    resolved = safe(lambda: measure.cov_estimator_)
    if resolved is not None:
        run.log_param(f"{prefix}.cov_estimator_resolved", type(resolved).__name__)
        if safe(lambda: measure.get_params()["cov_estimator"]) is None:
            run.log_param(f"{prefix}.cov_estimator.was_default", True)


def connectivity_fit_transform(measure: Any, timeseries: Any, run: Run,
                               prefix: str = "connectivity", **kwargs: Any):
    """``ConnectivityMeasure.fit_transform()`` with configuration and summary."""
    out = measure.fit_transform(timeseries, **kwargs)
    # Recorded after the fit: cov_estimator_ does not exist before it.
    describe_connectivity(measure, run, prefix)

    def _stats():
        import numpy as np
        arr = np.asarray(out)
        finite = arr[np.isfinite(arr)]
        return {
            "shape": list(map(int, arr.shape)),
            "mean": round(float(finite.mean()), 8) if finite.size else 0.0,
            "abs_max": round(float(np.abs(finite).max()), 8) if finite.size else 0.0,
            "n_nonfinite": int((~np.isfinite(arr)).sum()),
        }

    for key, value in (safe(_stats, {}) or {}).items():
        run.log_result(f"{prefix}.{key}", value)
    return out


# --------------------------------------------------------------------------
# GLM
# --------------------------------------------------------------------------

def describe_glm(model: Any, run: Run, prefix: str = "glm") -> None:
    """Record a first- or second-level model and its design matrix.

    The design matrix is the model. Its column names encode the conditions, the
    drift regressors and the confounds all at once, and two analyses with the
    same `hrf_model` and different designs are different experiments.
    """
    describe_estimator(model, run, prefix)

    designs = safe(lambda: list(model.design_matrices_), [])
    if not designs:
        design = safe(lambda: model.design_matrix_)
        designs = [design] if design is not None else []

    if designs:
        run.log_param(f"{prefix}.n_runs", len(designs))
        first = designs[0]
        columns = safe(lambda: [str(c) for c in first.columns], [])
        if columns:
            run.log_param(f"{prefix}.design_columns", columns)
            run.log_param(f"{prefix}.n_design_columns", len(columns))
            run.log_param(f"{prefix}.design_sha256", _sha(columns))
        run.log_param(f"{prefix}.design_shape",
                      safe(lambda: list(map(int, first.shape)), "unknown"))


def describe_contrast(name: str, stat_map: Any, run: Run,
                      prefix: str = "contrast") -> None:
    """Summarise a contrast map as comparable scalars, not an image."""
    run.log_param(f"{prefix}.{name}.computed", True)

    def _stats():
        import numpy as np
        data = np.asarray(stat_map.get_fdata())
        finite = data[np.isfinite(data)]
        return {
            "max": round(float(finite.max()), 6) if finite.size else 0.0,
            "min": round(float(finite.min()), 6) if finite.size else 0.0,
            "mean": round(float(finite.mean()), 8) if finite.size else 0.0,
            "n_nonfinite": int((~np.isfinite(data)).sum()),
        }

    for key, value in (safe(_stats, {}) or {}).items():
        run.log_result(f"{prefix}.{name}.{key}", value)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def describe_environment(run: Run, prefix: str = "nilearn") -> None:
    """Record nilearn and nibabel versions.

    The nilearn version matters more than usual here: the meaning of
    ``standardize=True`` changed between releases, so the parameter value alone
    does not identify what was done to the data.
    """
    from importlib.metadata import version

    run.log_param(f"{prefix}.version", safe(lambda: version("nilearn"), "unknown"))
    run.log_param(f"{prefix}.nibabel_version",
                  safe(lambda: version("nibabel"), "unknown"))


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever nilearn object is handed in."""
    cls = type(obj).__name__
    if "Masker" in cls:
        describe_masker(obj, run, prefix or "masker")
    elif "ConnectivityMeasure" in cls:
        describe_connectivity(obj, run, prefix or "connectivity")
    elif "LevelModel" in cls:
        describe_glm(obj, run, prefix or "glm")
    elif hasattr(obj, "affine"):
        describe_image(obj, run, prefix or "image")
    else:
        describe_estimator(obj, run, prefix or "estimator")
