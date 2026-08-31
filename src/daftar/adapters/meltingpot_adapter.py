"""MeltingPot adapter.

Chosen over Concordia for the first release for one reason: MeltingPot
substrates are deterministic given a seed, so a recorded run can genuinely be
reproduced. Concordia calls a language model on every agent step, and no major
provider guarantees token-level determinism even with a fixed seed -- so
``replay`` there can only ever mean "re-run and compare transcripts". That is a
worthwhile thing to build, and a bad thing to prove a concept with.

What a generic tracker misses about a MeltingPot run:

* **The substrate config is not the substrate name.** ``get_config`` returns a
  ConfigDict that callers routinely mutate before ``build_from_config``. The
  name alone loses that entirely, so the resolved config is hashed.
* **Roles determine the number of players and the game itself.** ``roles`` is a
  sequence whose length sets the player count; ``["default"] * 5`` and
  ``["default"] * 7`` are different experiments.
* **Bots are versioned.** A scenario pins pretrained bot policies; which bot
  checkpoints faced your agent is part of the result.
* **Episode length and lab2d settings.** ``maxEpisodeLengthFrames`` lives deep
  in the config and silently truncates returns.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

from ..run import Run
from .base import safe

name = "meltingpot"


def is_available() -> bool:
    try:
        import meltingpot  # noqa: F401
        return True
    except Exception:
        return False


def _config_hash(config: Any) -> str:
    """Stable hash of a ConfigDict, so mutation is detectable.

    Recording the whole config would swamp the manifest -- these run to
    thousands of lines of lua-facing settings. A hash plus the handful of fields
    people actually change is the right trade.
    """
    try:
        text = config.to_json(sort_keys=True)
    except Exception:
        try:
            text = repr(sorted(config.items()))
        except Exception:
            text = repr(config)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def describe_substrate_config(config: Any, run: Run, prefix: str = "substrate") -> None:
    run.log_param(f"{prefix}.config_sha256", _config_hash(config))

    for key in (
        "substrate", "num_players", "maxEpisodeLengthFrames",
        "default_player_roles", "valid_roles", "action_set",
    ):
        value = safe(lambda k=key: config[k])
        if value is None:
            continue
        if key in ("default_player_roles", "valid_roles"):
            run.log_param(f"{prefix}.{key}", sorted(map(str, value)))
        elif key == "action_set":
            run.log_param(f"{prefix}.n_actions", safe(lambda v=value: len(v), "unknown"))
        else:
            run.log_param(f"{prefix}.{key}", value)

    lab2d = safe(lambda: config["lab2d_settings"])
    if lab2d is not None:
        run.log_param(
            f"{prefix}.max_episode_frames",
            safe(lambda: lab2d["maxEpisodeLengthFrames"], "unknown"),
        )
        run.log_param(
            f"{prefix}.lab2d_seed", safe(lambda: lab2d["seed"], "unset"),
        )


def describe_substrate(
    substrate: Any, run: Run, *, name_: str | None = None,
    roles: Sequence[str] | None = None, prefix: str = "substrate",
) -> None:
    import meltingpot

    run.log_param(f"{prefix}.meltingpot_version",
                  safe(lambda: meltingpot.__version__, "unknown"))
    if name_:
        run.log_param(f"{prefix}.name", name_)
        cfg = safe(lambda: meltingpot.substrate.get_config(name_))
        if cfg is not None:
            describe_substrate_config(cfg, run, prefix)
    if roles is not None:
        run.log_param(f"{prefix}.roles", list(roles))
        run.log_param(f"{prefix}.n_players", len(roles))

    run.log_param(
        f"{prefix}.action_spec_n",
        safe(lambda: len(substrate.action_spec()), "unknown"),
    )
    obs = safe(lambda: substrate.observation_spec())
    if obs:
        run.log_param(
            f"{prefix}.observation_keys",
            safe(lambda: sorted(obs[0].keys()), "unknown"),
        )


def build(name_: str, roles: Sequence[str], run: Run):
    """``meltingpot.substrate.build`` with the resolved config recorded."""
    import meltingpot

    substrate = meltingpot.substrate.build(name_, roles=list(roles))
    describe_substrate(substrate, run, name_=name_, roles=roles)
    return substrate


def describe_scenario(scenario_name: str, run: Run, prefix: str = "scenario") -> None:
    """Record which pretrained bots a scenario pins.

    Bot checkpoints are part of the experiment. A scenario re-released with
    retrained bots produces different numbers from identical agent code, and
    without this field nothing in the record would show it.
    """
    import meltingpot

    run.log_param(f"{prefix}.name", scenario_name)
    cfg = safe(lambda: meltingpot.scenario.get_config(scenario_name))
    if cfg is None:
        return

    run.log_param(f"{prefix}.substrate", safe(lambda: str(cfg.substrate), "unknown"))
    run.log_param(f"{prefix}.roles", safe(lambda: list(map(str, cfg.roles)), []))
    run.log_param(f"{prefix}.tags", safe(lambda: sorted(map(str, cfg.tags)), []))

    # is_focal is a per-slot boolean sequence; the focal/bot split is the
    # experiment. n_focal is what people report, so record it directly rather
    # than making a reader count booleans.
    is_focal = safe(lambda: list(cfg.is_focal), [])
    if is_focal:
        run.log_param(f"{prefix}.is_focal", [bool(x) for x in is_focal])
        run.log_param(f"{prefix}.n_focal", sum(1 for x in is_focal if x))
        run.log_param(f"{prefix}.n_bots", sum(1 for x in is_focal if not x))

    # bots_by_role maps a role to the *set* of bots sampled for that role, so
    # flatten it. Which bot checkpoints faced your agent is part of the result:
    # a scenario re-released with retrained bots moves every number.
    def _bots():
        out = set()
        for role, names in cfg.bots_by_role.items():
            for b in names:
                out.add(f"{role}:{b}")
        return sorted(out)

    bots = safe(_bots, [])
    if bots:
        run.log_param(f"{prefix}.bots", bots)
        run.log_param(
            f"{prefix}.bots_sha256",
            hashlib.sha256("\n".join(bots).encode()).hexdigest()[:12],
        )


def run_episode(
    substrate: Any,
    policy: Callable[[Any, int], Any],
    run: Run,
    *,
    max_steps: int = 1000,
    prefix: str = "episode",
) -> dict[str, Any]:
    """Roll out one episode, recording per-player returns.

    Per-player returns rather than only the total: MeltingPot exists to study
    distributional outcomes, so a substrate where everyone scores 10 and one
    where a single player takes 70 have the same sum and are opposite findings.
    """
    timestep = substrate.reset()
    n_players = len(substrate.action_spec())
    returns = [0.0] * n_players
    steps = 0

    while not timestep.last() and steps < max_steps:
        actions = [policy(timestep, i) for i in range(n_players)]
        timestep = substrate.step(actions)
        rewards = timestep.reward
        if rewards is not None:
            for i, r in enumerate(rewards):
                returns[i] += float(r)
        steps += 1

    total = sum(returns)
    stats: dict[str, Any] = {
        "steps": steps,
        "truncated": steps >= max_steps and not timestep.last(),
        "total_return": round(total, 4),
        "mean_return": round(total / max(n_players, 1), 4),
        "min_return": round(min(returns), 4) if returns else 0.0,
        "max_return": round(max(returns), 4) if returns else 0.0,
    }

    def _gini():
        # Equality is usually the point of the experiment in this family of
        # substrates, so it belongs in the manifest rather than a notebook.
        vals = sorted(max(r, 0.0) for r in returns)
        n = len(vals)
        s = sum(vals)
        if n == 0 or s == 0:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(vals))
        return round((2 * cum) / (n * s) - (n + 1) / n, 4)

    stats["gini"] = safe(_gini, "unavailable")

    for k, v in stats.items():
        run.log_result(f"{prefix}.{k}", v)
    for i, r in enumerate(returns):
        run.log_result(f"{prefix}.return.player_{i}", round(r, 4))

    return {"returns": returns, **stats}


def describe(obj: Any, run: Run, prefix: str = "substrate") -> None:
    if isinstance(obj, str):
        describe_scenario(obj, run)
    else:
        describe_substrate(obj, run, prefix=prefix)
