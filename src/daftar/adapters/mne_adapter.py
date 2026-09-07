"""MNE-Python adapter.

MNE preprocessing is largely a sequence of **human decisions that are never
written down**. Which channels you marked bad, which ICA components you excluded
after looking at the topographies, which epochs were dropped and why: each one
changes every downstream number, and each one typically survives only in the
analyst's memory and a `.fif` file.

What this records that a generic tracker cannot:

* **`ica.exclude`.** The single most consequential unrecorded decision in EEG/MEG
  analysis. A reviewer asking "which components did you remove?" is asking a
  question that usually has no answer six months later.
* **Whether ICA converged.** `n_iter_ == max_iter` means FastICA hit the
  iteration limit and stopped, not that it finished. MNE reports this as a
  warning at fit time and stores nothing you would notice afterwards.
* **`random_state=None`.** ICA without a seed is not reproducible, and the
  resulting components -- and therefore the exclusions -- differ between runs.
* **Filter design parameters.** `raw.filter()` updates `info['highpass']` and
  `info['lowpass']` and then discards `fir_design`, `phase`, `window` and the
  transition bandwidths. Two analyses filtered "1--40 Hz" can differ
  substantially. :func:`filter_raw` records them.
* **`epochs.drop_log`.** How many epochs were rejected and for what reason. An
  evoked average over 40 surviving epochs is a different quantity from one over
  180, and the average alone does not say which you have.

**On subject data.** This adapter runs against human neuroimaging recordings, so
it records structure and never content: `subject_info` and file paths are
hashed, not stored, and `meas_date` is recorded only as present or absent
because dates of service are themselves identifiers. Manifests get committed to
public repositories; nothing here should make that a mistake.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "mne"


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("mne")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def _round(value: Any, digits: int = 6) -> Any:
    return safe(lambda: round(float(value), digits), value)


# --------------------------------------------------------------------------
# Info
# --------------------------------------------------------------------------

def describe_info(info: Any, run: Run, prefix: str = "recording") -> None:
    """Record acquisition and preprocessing state from an ``mne.Info``.

    Note what is *not* here: no subject identifiers, no measurement date, no
    file paths. See the module docstring.
    """
    run.log_param(f"{prefix}.sfreq", _round(safe(lambda: info["sfreq"])))
    run.log_param(f"{prefix}.nchan", safe(lambda: int(info["nchan"]), "unknown"))

    # The filter band as MNE currently believes it to be. The *design* that
    # produced it is not in Info; see filter_raw().
    run.log_param(f"{prefix}.highpass", _round(safe(lambda: info["highpass"])))
    run.log_param(f"{prefix}.lowpass", _round(safe(lambda: info["lowpass"])))
    run.log_param(f"{prefix}.line_freq", safe(lambda: info["line_freq"], "unset"))

    # Bad channels are a human judgement that silently changes every average,
    # every ICA decomposition and every source estimate downstream.
    bads = safe(lambda: sorted(info["bads"]), [])
    run.log_param(f"{prefix}.bads", bads)
    run.log_param(f"{prefix}.n_bads", len(bads))

    ref = safe(lambda: info["custom_ref_applied"])
    if ref is not None:
        run.log_param(f"{prefix}.custom_ref_applied", str(ref))

    projs = safe(lambda: list(info["projs"]), [])
    run.log_param(f"{prefix}.n_projs", len(projs))
    if projs:
        run.log_param(
            f"{prefix}.projs",
            safe(lambda: sorted(f"{p['desc']}({'on' if p['active'] else 'off'})"
                                for p in projs), []),
        )

    dig = safe(lambda: info["dig"])
    run.log_param(f"{prefix}.has_montage", dig is not None)
    if dig is not None:
        run.log_param(f"{prefix}.n_dig_points", safe(lambda: len(dig), "unknown"))

    # Present-or-absent only. A date of service is an identifier.
    run.log_param(f"{prefix}.meas_date_set",
                  safe(lambda: info["meas_date"], None) is not None)

    subject = safe(lambda: info["subject_info"])
    if subject:
        # Hashed so a changed or swapped subject is detectable without the
        # manifest carrying anything about who they are.
        run.log_param(f"{prefix}.subject_info_sha256",
                      _sha(sorted(map(str, subject.items()))))


def _channel_type_counts(obj: Any) -> dict[str, int]:
    def _counts():
        from collections import Counter
        return dict(Counter(obj.get_channel_types()))
    return safe(_counts, {}) or {}


# --------------------------------------------------------------------------
# Raw
# --------------------------------------------------------------------------

def describe_raw(raw: Any, run: Run, prefix: str = "recording") -> None:
    """Record a Raw object: info, duration, channel composition, annotations."""
    run.log_param(f"{prefix}.type", type(raw).__name__)
    describe_info(safe(lambda: raw.info), run, prefix)

    run.log_param(f"{prefix}.n_times", safe(lambda: int(raw.n_times), "unknown"))
    run.log_param(f"{prefix}.duration_s",
                  _round(safe(lambda: raw.times[-1]), 3))
    run.log_param(f"{prefix}.preload", bool(safe(lambda: raw.preload, False)))

    for ch_type, count in sorted(_channel_type_counts(raw).items()):
        run.log_param(f"{prefix}.n_{ch_type}", count)

    # File paths can contain subject identifiers, so hash them. The count still
    # distinguishes a single recording from a concatenation.
    files = safe(lambda: [str(f) for f in raw.filenames], [])
    if files:
        run.log_param(f"{prefix}.n_source_files", len(files))
        run.log_param(f"{prefix}.source_files_sha256", _sha("\n".join(sorted(files))))

    ann = safe(lambda: raw.annotations)
    if ann is not None:
        run.log_param(f"{prefix}.n_annotations", safe(lambda: len(ann), 0))
        descriptions = safe(lambda: sorted(set(map(str, ann.description))), [])
        if descriptions:
            run.log_param(f"{prefix}.annotation_kinds", descriptions)


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

#: ``raw.filter`` defaults, recorded explicitly when not passed. MNE stores the
#: resulting band in ``info`` and discards the design, so a future change of
#: default would otherwise be invisible.
FILTER_DEFAULTS = {
    "method": "fir",
    "fir_design": "firwin",
    "fir_window": "hamming",
    "phase": "zero",
    "l_trans_bandwidth": "auto",
    "h_trans_bandwidth": "auto",
    "filter_length": "auto",
    "pad": "reflect_limited",
}


def filter_raw(raw: Any, run: Run, l_freq=None, h_freq=None,
               prefix: str = "filter", **kwargs: Any):
    """``raw.filter()`` with the filter *design* recorded, not just the band.

    ::

        with daftar.track("preproc", seed=42) as run:
            mne_adapter.filter_raw(raw, run, l_freq=1.0, h_freq=40.0)

    After a plain ``raw.filter(1, 40)``, ``info['highpass']`` and
    ``info['lowpass']`` hold 1 and 40 and everything else is gone. But a
    zero-phase FIR with a 0.5 Hz transition band and a causal IIR are different
    filters, and "1--40 Hz" in a methods section does not distinguish them.
    """
    resolved = dict(FILTER_DEFAULTS)
    resolved.update(kwargs)

    run.log_param(f"{prefix}.l_freq", l_freq if l_freq is not None else "none")
    run.log_param(f"{prefix}.h_freq", h_freq if h_freq is not None else "none")
    for key, value in sorted(resolved.items()):
        run.log_param(f"{prefix}.{key}", value)
        if key in FILTER_DEFAULTS and key not in kwargs:
            run.log_param(f"{prefix}.{key}.was_default", True)

    out = raw.filter(l_freq=l_freq, h_freq=h_freq, **kwargs)
    describe_info(safe(lambda: raw.info), run, "recording")
    return out


def describe_resample(raw: Any, run: Run, old_sfreq: float,
                      prefix: str = "resample") -> None:
    """Record a resampling step, which is lossy and not reversible."""
    run.log_param(f"{prefix}.sfreq_before", _round(old_sfreq))
    run.log_param(f"{prefix}.sfreq_after",
                  _round(safe(lambda: raw.info["sfreq"])))


# --------------------------------------------------------------------------
# Epochs
# --------------------------------------------------------------------------

def describe_epochs(epochs: Any, run: Run, prefix: str = "epochs") -> None:
    """Record epoching parameters and, importantly, what was thrown away.

    ``drop_log`` is the interesting part. An evoked average over 40 surviving
    epochs is a different quantity from one over 180, and the average alone does
    not say which you are looking at. The reasons matter too: epochs dropped for
    amplitude rejection and epochs dropped because the event was missing are
    different problems.
    """
    run.log_param(f"{prefix}.type", type(epochs).__name__)
    run.log_param(f"{prefix}.tmin", _round(safe(lambda: epochs.tmin)))
    run.log_param(f"{prefix}.tmax", _round(safe(lambda: epochs.tmax)))
    run.log_param(f"{prefix}.baseline", str(safe(lambda: epochs.baseline, "none")))
    run.log_param(f"{prefix}.reject", safe(lambda: epochs.reject, "none") or "none")
    run.log_param(f"{prefix}.flat", safe(lambda: epochs.flat, "none") or "none")
    run.log_param(f"{prefix}.decim", safe(lambda: getattr(epochs, "decim", 1), 1))
    run.log_param(f"{prefix}.detrend", safe(lambda: epochs.detrend, "none"))

    event_id = safe(lambda: dict(epochs.event_id), {})
    if event_id:
        run.log_param(f"{prefix}.event_id", event_id)
        run.log_param(f"{prefix}.n_conditions", len(event_id))

    n_kept = safe(lambda: len(epochs), 0)
    run.log_result(f"{prefix}.n_epochs_kept", n_kept)

    drop_log = safe(lambda: epochs.drop_log)
    if drop_log is not None:
        total = len(drop_log)
        dropped = [d for d in drop_log if d]
        run.log_result(f"{prefix}.n_epochs_total", total)
        run.log_result(f"{prefix}.n_epochs_dropped", len(dropped))
        if total:
            run.log_result(f"{prefix}.drop_rate",
                           round(len(dropped) / total, 4))

        def _reasons():
            from collections import Counter
            counts = Counter()
            for entry in dropped:
                for reason in entry:
                    counts[str(reason)] += 1
            return dict(counts.most_common())

        reasons = safe(_reasons, {})
        if reasons:
            run.log_result(f"{prefix}.drop_reasons", reasons)


def describe_evoked(evoked: Any, run: Run, prefix: str = "evoked") -> None:
    """Summarise an evoked response as comparable scalars, not a waveform."""
    run.log_result(f"{prefix}.n_averaged", safe(lambda: int(evoked.nave), "unknown"))
    run.log_result(f"{prefix}.comment", safe(lambda: str(evoked.comment), ""))

    def _stats():
        import numpy as np
        data = np.asarray(evoked.data)
        return {
            "peak_abs": float(np.abs(data).max()),
            "mean": float(data.mean()),
            "rms": float(np.sqrt((data ** 2).mean())),
            "n_nonfinite": int((~np.isfinite(data)).sum()),
        }

    for key, value in (safe(_stats, {}) or {}).items():
        run.log_result(f"{prefix}.{key}",
                       value if key == "n_nonfinite" else round(value, 12))


# --------------------------------------------------------------------------
# ICA -- the reason this adapter exists
# --------------------------------------------------------------------------

def describe_ica(ica: Any, run: Run, prefix: str = "ica") -> None:
    """Record an ICA decomposition, including the exclusions and convergence.

    ``ica.exclude`` is the most consequential unrecorded decision in EEG/MEG
    preprocessing: a person looked at topographies and time courses and chose
    which components were artefact. Nothing downstream records that choice, and
    a reviewer asking "which components did you remove?" is usually asking a
    question with no surviving answer.
    """
    run.log_param(f"{prefix}.method", safe(lambda: str(ica.method), "unknown"))
    run.log_param(f"{prefix}.n_components_requested",
                  safe(lambda: ica.n_components, "auto"))
    run.log_param(f"{prefix}.n_components_fitted",
                  safe(lambda: int(ica.n_components_), "unknown"))
    run.log_param(f"{prefix}.max_iter", safe(lambda: ica.max_iter, "unknown"))

    # random_state=None means the decomposition is not reproducible, and neither
    # are the component indices the exclusions refer to.
    random_state = safe(lambda: ica.random_state)
    run.log_param(f"{prefix}.random_state",
                  "none (NOT REPRODUCIBLE)" if random_state is None else random_state)

    fit_params = safe(lambda: dict(ica.fit_params or {}), {})
    for key, value in sorted(fit_params.items()):
        run.log_param(f"{prefix}.fit_params.{key}", value)

    n_iter = safe(lambda: int(ica.n_iter_))
    max_iter = safe(lambda: ica.max_iter)
    if n_iter is not None:
        run.log_result(f"{prefix}.n_iter", n_iter)
        # Hitting the iteration limit means it stopped, not that it finished.
        if isinstance(max_iter, int):
            run.log_result(f"{prefix}.converged", n_iter < max_iter)

    run.log_param(f"{prefix}.n_channels_used",
                  safe(lambda: len(ica.ch_names), "unknown"))
    run.log_param(f"{prefix}.n_samples", safe(lambda: int(ica.n_samples_), "unknown"))

    exclude = sorted(safe(lambda: [int(i) for i in ica.exclude], []) or [])
    run.log_param(f"{prefix}.exclude", exclude)
    run.log_param(f"{prefix}.n_excluded", len(exclude))


def apply_ica(ica: Any, inst: Any, run: Run, prefix: str = "ica", **kwargs: Any):
    """``ica.apply()`` with the decomposition and exclusions recorded first.

    ::

        with daftar.track("preproc", seed=42) as run:
            ica.exclude = [0, 3]
            mne_adapter.apply_ica(ica, raw, run)

    Recording happens *before* the call, so the exclusion list that was actually
    applied is what lands in the manifest even if it is edited afterwards.
    """
    describe_ica(ica, run, prefix)
    for key, value in sorted(kwargs.items()):
        run.log_param(f"{prefix}.apply.{key}", value)
    return ica.apply(inst, **kwargs)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def describe_environment(run: Run, prefix: str = "mne") -> None:
    import mne

    run.log_param(f"{prefix}.version", safe(lambda: mne.__version__, "unknown"))


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever MNE object is handed in."""
    cls = type(obj).__name__
    if cls == "ICA":
        describe_ica(obj, run, prefix or "ica")
    elif "Epochs" in cls:
        describe_epochs(obj, run, prefix or "epochs")
    elif cls in ("Evoked", "EvokedArray"):
        describe_evoked(obj, run, prefix or "evoked")
    elif cls == "Info":
        describe_info(obj, run, prefix or "recording")
    else:
        describe_raw(obj, run, prefix or "recording")
