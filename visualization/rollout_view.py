"""
Generate a same-seed rollout comparison for bursty Poisson traffic.

Run from project root:
    python visualization/rollout_view.py

Saves:
    experiments/figures/poisson_burst_rollout_seed0.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.conflict_aware import conflict_aware_policy
from baselines.runway_aware import runway_aware_policy
from env.airport_env import AirportEnv
from evaluation.evaluate import episode_metrics
from simulator import AirportSimulator

OUT_DIR = "experiments/figures"
SCENARIO = "poisson_burst"
SEED = 0


@dataclass
class Trace:
    name: str
    steps: list[int]
    backlog: list[int]
    active: list[int]
    completed: list[int]
    generated: list[int]
    runway_queue: list[int]
    cumulative_reward: list[float]
    short_routes: list[int]
    bypass_routes: list[int]
    final_metrics: dict
    total_reward: float
    timeout: bool


def _record_state(trace: Trace, state: dict, cumulative_reward: float) -> None:
    trace.steps.append(int(state["step"]))
    trace.backlog.append(int(state.get("backlog_count", 0)))
    trace.active.append(int(state.get("active_unfinished_count", state.get("n_active", 0))))
    trace.completed.append(len(state.get("completed_records", [])))
    trace.generated.append(int(state.get("generated_departures", 0) + state.get("generated_arrivals", 0)))
    trace.runway_queue.append(int(state.get("runway_queue_len", 0)))
    trace.cumulative_reward.append(float(cumulative_reward))
    trace.short_routes.append(int(state.get("short_route_count", 0)))
    trace.bypass_routes.append(int(state.get("bypass_route_count", 0)))


def run_baseline_trace(name: str, policy_fn: Callable[[dict], int]) -> Trace:
    sim = AirportSimulator(scenario=SCENARIO)
    state = sim.reset(seed=SEED)
    trace = Trace(name, [], [], [], [], [], [], [], [], [], {}, 0.0, False)
    cumulative_reward = 0.0
    _record_state(trace, state, cumulative_reward)

    done = False
    last_info = {}
    while not done:
        state["_sim"] = sim
        action = int(policy_fn(state))
        state.pop("_sim", None)
        state, reward, done, last_info = sim.step(action)
        cumulative_reward += float(reward)
        _record_state(trace, state, cumulative_reward)

    trace.final_metrics = episode_metrics(sim)
    trace.total_reward = cumulative_reward
    trace.timeout = bool(last_info.get("timeout", False))
    return trace


def run_maskable_ppo_trace(name: str, checkpoint: str, strategic_noop: bool = False) -> Trace:
    env = AirportEnv(
        scenario=SCENARIO,
        obs_mode="poisson_features",
        strategic_noop=strategic_noop,
    )
    model = MaskablePPO.load(checkpoint)
    obs, _ = env.reset(seed=SEED)
    state = env._sim._get_state()
    trace = Trace(name, [], [], [], [], [], [], [], [], [], {}, 0.0, False)
    cumulative_reward = 0.0
    _record_state(trace, state, cumulative_reward)

    done = False
    last_info = {}
    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        obs, reward, terminated, truncated, last_info = env.step(int(action))
        done = terminated or truncated
        cumulative_reward += float(reward)
        _record_state(trace, env._sim._get_state(), cumulative_reward)

    trace.final_metrics = episode_metrics(env._sim)
    trace.total_reward = cumulative_reward
    trace.timeout = bool(last_info.get("timeout", False))
    env.close()
    return trace


def _style(ax) -> None:
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _shade_burst_phases(ax) -> None:
    ax.axvspan(0, 20, color="#d6eaf8", alpha=0.45, linewidth=0)
    ax.axvspan(20, 85, color="#fadbd8", alpha=0.45, linewidth=0)
    ax.axvspan(85, 200, color="#d5f5e3", alpha=0.35, linewidth=0)
    ymax = ax.get_ylim()[1]
    ax.text(10, ymax, "low", ha="center", va="top", fontsize=8, color="#34495e")
    ax.text(52.5, ymax, "rush", ha="center", va="top", fontsize=8, color="#7b241c")
    ax.text(142.5, ymax, "recovery", ha="center", va="top", fontsize=8, color="#145a32")


def plot_traces(traces: list[Trace]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    colors = {
        "ConflictAware":   "#27ae60",
        "RunwayAware":     "#16a085",
        "PPO features":    "#d35400",
        "PPO features+hold": "#154360",
    }

    # 2×2 grid matching the flat layout
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    fig.patch.set_facecolor("white")

    ax_backlog  = axes[0][0]
    ax_complete = axes[0][1]
    ax_reward   = axes[1][0]
    ax_runway   = axes[1][1]

    for ax in [ax_backlog, ax_complete, ax_reward, ax_runway]:
        _style(ax)

    for trace in traces:
        color = colors.get(trace.name, "#555")
        steps = np.array(trace.steps)
        lw = 2.0
        ax_backlog.plot(steps,  trace.backlog,            color=color, linewidth=lw, label=trace.name)
        ax_complete.plot(steps, trace.completed,           color=color, linewidth=lw, label=trace.name)
        ax_reward.plot(steps,   trace.cumulative_reward,   color=color, linewidth=lw, label=trace.name)
        ax_runway.plot(steps,   trace.runway_queue,        color=color, linewidth=lw, label=trace.name)

    ax_backlog.set_ylabel("Backlog",            fontsize=10)
    ax_complete.set_ylabel("Completed demand",  fontsize=10)
    ax_reward.set_ylabel("Cumulative reward",   fontsize=10)
    ax_runway.set_ylabel("Runway queue",        fontsize=10)

    for ax in [ax_reward, ax_runway]:
        ax.set_xlabel("Timestep", fontsize=10)

    ax_runway.set_yticks([0, 1])
    ax_reward.axhline(0, color="#333", linewidth=0.8, linestyle="--", alpha=0.5)

    for ax in [ax_backlog, ax_complete, ax_reward, ax_runway]:
        _shade_burst_phases(ax)

    handles, labels = ax_backlog.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(traces),
        fontsize=9,
        framealpha=0.95,
        edgecolor="#ccc",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    # Save to same filename as the committed figure so it replaces it
    out_path = os.path.join(OUT_DIR, "poisson_burst_report_rollout_flat.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    traces = [
        run_baseline_trace("ConflictAware", conflict_aware_policy),
        run_baseline_trace("RunwayAware", runway_aware_policy),
        run_maskable_ppo_trace(
            "PPO features",
            "experiments/models/ppo_poisson_mix_features_maskable_seed0.zip",
        ),
        run_maskable_ppo_trace(
            "PPO features+hold",
            "experiments/models/ppo_poisson_mix_features_hold_long_maskable_seed0.zip",
            strategic_noop=True,
        ),
    ]
    out_path = plot_traces(traces)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
