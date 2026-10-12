"""Concordia adapter — provenance for LLM-driven simulation.

This adapter has a **different contract** from every other one in this package,
and the difference is the point.

Everywhere else, daftar promises that a recorded run can be rebuilt: same code,
same seed, same inputs, same answer. Concordia cannot honour that. Every agent
step calls ``LanguageModel.sample_text()``, and no major provider guarantees
token-level determinism even with a fixed seed — temperature, batching,
load-balanced serving and silent model updates all break it. Shipping an adapter
whose ``replay`` silently did not replay would undermine the one property this
package sells, so this adapter does not claim reproducibility at all.

**What it claims instead: auditability.** It wraps the language model, records a
hash of every ``(prompt, response)`` pair in call order, and can tell you the
**first step at which two runs diverged**. That is a strictly weaker guarantee
and an honest one — and for LLM-driven simulation it is more than anything else
currently offers, because nobody can presently audit these runs at all.

Concretely, comparing two runs of the same scenario answers questions that are
otherwise unanswerable:

* Did the two runs follow identical trajectories, or did they split?
* If they split, at which call — step 3, or step 147?
* Was the divergence in what was *asked* (prompt hash differs: the simulation
  state had already diverged) or in what came *back* (prompt identical, response
  differs: the model itself was non-deterministic at that point)?

That last distinction is the useful one. It separates "my simulation is
non-deterministic" from "the provider gave me a different answer to the same
question", and those have completely different remedies.

**Transcripts are hashed, not stored.** Prompts in agent simulations routinely
contain the entire scenario, and responses can be long. The manifest gets
fixed-width hashes; :func:`write_transcript` writes the full text to a file that
the export bundle carries, so the content is available without bloating a
record meant to be committed to git.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Collection, Mapping, Sequence

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "concordia"

#: Concordia's sampling defaults, recorded when the caller does not override
#: them. Temperature especially: 1.0 is the default and is *not* a setting most
#: people would choose deliberately for a simulation they intend to compare.
SAMPLING_DEFAULTS = {
    "max_tokens": 5000,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "timeout": 60,
}


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("concordia")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _h(text: str, n: int = 8) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:n]


# --------------------------------------------------------------------------
# the recording wrapper
# --------------------------------------------------------------------------

class RecordingLanguageModel:
    """Wraps a Concordia ``LanguageModel`` and records every call.

    Delegates everything to the wrapped model, so it is a drop-in replacement:

    ::

        model = concordia_adapter.wrap(real_model, run)
        # ... build and run the simulation exactly as before ...

    Deliberately *not* a subclass of ``LanguageModel``. Subclassing would mean
    tracking Concordia's abstract interface as it changes and breaking whenever
    a method is added; delegating by attribute lookup means anything this class
    does not implement passes straight through to the wrapped model.
    """

    def __init__(self, model: Any, run: Run, prefix: str = "llm",
                 store_transcript: bool = True):
        self._model = model
        self._run = run
        self._prefix = prefix
        self._store_transcript = store_transcript

        #: Ordered ``(prompt_hash, response_hash)`` pairs. The chain that lets
        #: two runs be compared step by step.
        self.calls: list[dict[str, Any]] = []
        #: Running chain hash: each entry depends on every call before it, so a
        #: single value identifies the whole trajectory.
        self._chain = hashlib.sha256()
        self._transcript: list[dict[str, Any]] = []

    # -- delegation ------------------------------------------------------

    def __getattr__(self, item: str) -> Any:
        # Only reached for attributes this class does not define.
        return getattr(self._model, item)

    # -- the recorded calls ----------------------------------------------

    def sample_text(self, prompt: str, **kwargs: Any) -> str:
        response = self._model.sample_text(prompt, **kwargs)
        self._record("sample_text", prompt, response, kwargs)
        return response

    def sample_choice(self, prompt: str, responses: Sequence[str],
                      **kwargs: Any) -> tuple[int, str, Mapping[str, Any]]:
        idx, response, info = self._model.sample_choice(prompt, responses, **kwargs)
        self._record("sample_choice", prompt, response, kwargs,
                     extra={"n_options": len(responses),
                            "options_sha256": _h(json.dumps(list(responses),
                                                            sort_keys=True)),
                            "chosen_index": idx})
        return idx, response, info

    def _record(self, method: str, prompt: str, response: str,
                kwargs: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        prompt_hash = _h(prompt)
        response_hash = _h(response)

        # Chain each call into the running hash, so the trajectory hash depends
        # on order as well as content.
        self._chain.update(prompt_hash.encode())
        self._chain.update(b"\x00")
        self._chain.update(response_hash.encode())
        self._chain.update(b"\n")

        entry = {
            "i": len(self.calls),
            "method": method,
            "prompt_sha256": prompt_hash,
            "response_sha256": response_hash,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "seed": kwargs.get("seed"),
            "temperature": kwargs.get("temperature"),
        }
        if extra:
            entry.update(extra)
        self.calls.append(entry)

        if self._store_transcript:
            self._transcript.append({**entry, "prompt": prompt,
                                     "response": response})

    # -- reporting -------------------------------------------------------

    @property
    def trajectory_hash(self) -> str:
        return self._chain.hexdigest()[:16]

    def describe(self) -> None:
        """Write the recorded call statistics into the run's manifest."""
        run, prefix = self._run, self._prefix
        calls = self.calls

        run.log_param(f"{prefix}.model", type(self._model).__name__)
        for attr in ("model_name", "_model_name", "model"):
            value = safe(lambda a=attr: getattr(self._model, a))
            if isinstance(value, str):
                run.log_param(f"{prefix}.model_name", value)
                break

        run.log_result(f"{prefix}.n_calls", len(calls))
        if not calls:
            return

        # The trajectory: one value identifying the whole ordered exchange.
        run.log_result(f"{prefix}.trajectory_sha256", self.trajectory_hash)

        # The per-call chain, compact enough for a manifest and exactly what
        # first_divergence() needs to locate where two runs split.
        run.log_result(f"{prefix}.prompt_hashes",
                       [c["prompt_sha256"] for c in calls])
        run.log_result(f"{prefix}.response_hashes",
                       [c["response_sha256"] for c in calls])

        methods = sorted({c["method"] for c in calls})
        run.log_param(f"{prefix}.methods_used", methods)
        run.log_result(f"{prefix}.n_sample_text",
                       sum(1 for c in calls if c["method"] == "sample_text"))
        run.log_result(f"{prefix}.n_sample_choice",
                       sum(1 for c in calls if c["method"] == "sample_choice"))

        run.log_result(f"{prefix}.total_prompt_chars",
                       sum(c["prompt_chars"] for c in calls))
        run.log_result(f"{prefix}.total_response_chars",
                       sum(c["response_chars"] for c in calls))

        # Sampling settings. Concordia's default temperature is 1.0, which is
        # not what most people would choose for a run they intend to compare.
        temperatures = {c["temperature"] for c in calls}
        resolved = {t if t is not None else SAMPLING_DEFAULTS["temperature"]
                    for t in temperatures}
        run.log_param(f"{prefix}.temperatures", sorted(map(str, resolved)))
        if temperatures == {None}:
            run.log_param(f"{prefix}.temperature_was_default", True)

        seeds = {c["seed"] for c in calls}
        if seeds == {None}:
            # The honest headline: this run cannot be repeated even in principle.
            run.log_param(f"{prefix}.seeds_passed", False)
            run.log_param(
                f"{prefix}.determinism",
                "no seed passed to the model; this run is not repeatable even "
                "in principle. daftar records the trajectory so two runs can be "
                "compared, not reproduced.",
            )
        else:
            run.log_param(f"{prefix}.seeds_passed", True)
            run.log_param(f"{prefix}.seeds",
                          sorted(str(s) for s in seeds if s is not None))
            run.log_param(
                f"{prefix}.determinism",
                "seeds were passed, but no major provider guarantees "
                "token-level determinism; treat identical trajectories as "
                "evidence, not proof.",
            )

    # -- transcript ------------------------------------------------------

    def write_transcript(self, path: str | os.PathLike) -> str:
        """Write the full prompts and responses to a JSON file.

        Kept out of the manifest deliberately: prompts in agent simulations
        routinely contain the entire scenario. The manifest gets hashes; this
        file carries the content, and :meth:`Run.add_output` registers it so the
        export bundle includes it and its hash.
        """
        path = str(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"trajectory_sha256": self.trajectory_hash,
                       "n_calls": len(self.calls),
                       "calls": self._transcript}, fh, indent=2,
                      ensure_ascii=False)
        self._run.add_output(path)
        return path


def wrap(model: Any, run: Run, prefix: str = "llm",
         store_transcript: bool = True) -> RecordingLanguageModel:
    """Wrap a Concordia language model so every call is recorded.

    ::

        with daftar.track("village-scenario", seed=42) as run:
            model = concordia_adapter.wrap(real_model, run)
            ...                       # build and run the simulation
            concordia_adapter.finish(model, run)
    """
    return RecordingLanguageModel(model, run, prefix=prefix,
                                  store_transcript=store_transcript)


def finish(model: RecordingLanguageModel, run: Run,
           transcript_path: str | os.PathLike | None = None) -> None:
    """Record the wrapped model's statistics, and optionally the transcript."""
    describe_environment(run)
    model.describe()
    if transcript_path is not None:
        model.write_transcript(transcript_path)


# --------------------------------------------------------------------------
# comparing two runs -- the capability this adapter exists for
# --------------------------------------------------------------------------

def _hash_list(manifest: Any, key: str) -> list[str]:
    raw = manifest.get(key)
    if not raw:
        return []
    return [part.strip() for part in raw.strip("[]").split(",") if part.strip()]


def first_divergence(manifest_a: Any, manifest_b: Any,
                     prefix: str = "llm") -> dict[str, Any]:
    """Find the first call at which two recorded runs took different paths.

    Returns a dict with ``diverged``, and when they did: ``step``, ``kind``, and
    the hashes involved.

    ``kind`` is the part worth reading:

    * ``"prompt"`` — the model was asked a *different question* at this step, so
      the simulation state had already diverged before the model was called. The
      cause is upstream: agent memory, ordering, or an earlier response.
    * ``"response"`` — the model was asked the *same question* and gave a
      different answer. That is provider non-determinism, and no amount of
      fixing your code will remove it.

    Those two have completely different remedies, and no other tool
    distinguishes them.
    """
    pa = _hash_list(manifest_a, f"result.{prefix}.prompt_hashes")
    ra = _hash_list(manifest_a, f"result.{prefix}.response_hashes")
    pb = _hash_list(manifest_b, f"result.{prefix}.prompt_hashes")
    rb = _hash_list(manifest_b, f"result.{prefix}.response_hashes")

    if not pa or not pb:
        return {"diverged": None,
                "reason": "one or both runs recorded no language-model calls"}

    for i in range(min(len(pa), len(pb))):
        if pa[i] != pb[i]:
            return {"diverged": True, "step": i, "kind": "prompt",
                    "a": pa[i], "b": pb[i], "n_calls_a": len(pa),
                    "n_calls_b": len(pb)}
        if ra[i] != rb[i]:
            return {"diverged": True, "step": i, "kind": "response",
                    "prompt": pa[i], "a": ra[i], "b": rb[i],
                    "n_calls_a": len(pa), "n_calls_b": len(pb)}

    if len(pa) != len(pb):
        return {"diverged": True, "step": min(len(pa), len(pb)),
                "kind": "length", "n_calls_a": len(pa), "n_calls_b": len(pb)}

    return {"diverged": False, "n_calls": len(pa)}


def render_divergence(result: dict[str, Any]) -> str:
    """Human-readable form of :func:`first_divergence`."""
    if result.get("diverged") is None:
        return result.get("reason", "no comparison possible")
    if not result["diverged"]:
        return (f"Identical trajectories across all {result['n_calls']} "
                f"language-model calls.")

    step, kind = result["step"], result["kind"]
    if kind == "prompt":
        return (
            f"Diverged at call {step}: the model was asked a DIFFERENT question "
            f"({result['a']} vs {result['b']}).\n"
            f"The simulation state had already diverged before this call -- look "
            f"upstream at agent memory, ordering, or an earlier response."
        )
    if kind == "response":
        return (
            f"Diverged at call {step}: the model was asked the SAME question "
            f"({result['prompt']}) and gave a different answer "
            f"({result['a']} vs {result['b']}).\n"
            f"This is provider non-determinism. No change to your code removes it."
        )
    return (f"Trajectories agree for {step} calls, then differ in length: "
            f"{result['n_calls_a']} vs {result['n_calls_b']}.")


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def describe_environment(run: Run, prefix: str = "concordia") -> None:
    from importlib.metadata import version

    for dist in ("gdm-concordia", "concordia"):
        found = safe(lambda d=dist: version(d))
        if found:
            run.log_param(f"{prefix}.version", found)
            break
    else:
        run.log_param(f"{prefix}.version", "unknown")


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch. A wrapped model reports itself; anything else is named."""
    if isinstance(obj, RecordingLanguageModel):
        obj.describe()
    else:
        run.log_param(f"{prefix or 'concordia'}.type", type(obj).__name__)
