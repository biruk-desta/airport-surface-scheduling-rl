"""
Create a presentation GIF for the bursty Poisson rollout.

The animation is intentionally explanatory, not a full simulator UI. It compares
the same demand seed under a strong heuristic and the final MaskablePPO policy.

Run from project root:
    python visualization/rollout_animation.py

Saves:
    experiments/figures/poisson_burst_runwayaware_vs_ppo_hold.gif
    experiments/figures/poisson_burst_runwayaware_vs_ppo_hold_frames/*.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.runway_aware import runway_aware_policy
from env.airport_env import AirportEnv
from evaluation.evaluate import episode_metrics
from simulator import (
    APRON,
    BYPASS_I,
    BYPASS_LINK,
    GATE_A,
    GATE_B,
    GATE_C,
    GATE_D,
    NODE_NAMES,
    RUNWAY_EXIT,
    RUNWAY_SERVICE,
    RUNWAY_THRESHOLD,
    SPOT_A,
    SPOT_B,
    INTERSECTION_2,
    AirportSimulator,
    decode_action,
    CHOICE_ADVANCE,
    CHOICE_BYPASS,
    CHOICE_SHORT,
)

OUT_DIR = "experiments/figures"
FRAMES_DIR = os.path.join(OUT_DIR, "poisson_burst_runwayaware_vs_ppo_hold_frames")
SCENARIO = "poisson_burst"
SEED = 0
FRAME_STRIDE = 1
INTERP_STEPS = 3
FPS = 6

PPO_HOLD_CHECKPOINT = (
    "experiments/models/ppo_poisson_mix_features_hold_long_maskable_seed1.zip"
)

NODE_POS = {
    GATE_A: (0.0, 3.0),
    GATE_B: (0.0, 2.0),
    GATE_C: (0.0, 1.05),
    GATE_D: (0.0, 0.05),
    SPOT_A: (1.4, 2.5),
    SPOT_B: (1.4, 0.55),
    INTERSECTION_2: (3.2, 1.8),
    BYPASS_I: (3.35, -0.25),
    BYPASS_LINK: (5.05, -0.25),
    RUNWAY_THRESHOLD: (5.85, 1.2),
    RUNWAY_SERVICE: (7.0, 1.2),
    RUNWAY_EXIT: (5.2, 3.0),
    APRON: (3.6, 3.0),
}

EDGES = [
    (GATE_A, SPOT_A),
    (GATE_B, SPOT_A),
    (GATE_C, SPOT_B),
    (GATE_D, SPOT_B),
    (SPOT_A, INTERSECTION_2),
    (SPOT_B, INTERSECTION_2),
    (SPOT_A, BYPASS_I),
    (SPOT_B, BYPASS_I),
    (BYPASS_I, BYPASS_LINK),
    (BYPASS_LINK, RUNWAY_THRESHOLD),
    (INTERSECTION_2, RUNWAY_THRESHOLD),
    (RUNWAY_THRESHOLD, RUNWAY_SERVICE),
    (RUNWAY_EXIT, INTERSECTION_2),
    (INTERSECTION_2, APRON),
]

NODE_LABELS = {
    GATE_A: "GA",
    GATE_B: "GB",
    GATE_C: "GC",
    GATE_D: "GD",
    SPOT_A: "Spot A",
    SPOT_B: "Spot B",
    INTERSECTION_2: "Central",
    BYPASS_I: "Bypass",
    BYPASS_LINK: "Bypass link",
    RUNWAY_THRESHOLD: "Queue",
    RUNWAY_SERVICE: "Runway",
    RUNWAY_EXIT: "Arr exit",
    APRON: "Apron",
}


@dataclass
class Frame:
    state: dict
    action_label: str
    action_kind: str
    cumulative_reward: float


@dataclass
class Rollout:
    name: str
    frames: list[Frame]
    final_metrics: dict


def phase_name(step: int) -> str:
    if step < 20:
        return "LOW"
    if step < 85:
        return "RUSH"
    return "RECOVERY"


def aircraft_label(ac: dict) -> str:
    prefix = "A" if "arrival" in ac["route_key"] else "D"
    global_id = ac.get("global_id", ac["id"])
    suffix = ""
    if ac["route_key"].endswith("_short"):
        suffix = "-S"
    elif ac["route_key"].endswith("_bypass"):
        suffix = "-B"
    return f"{prefix}{global_id}{suffix}"


def describe_action(state: dict, action: int) -> tuple[str, str]:
    if action == 0:
        return "HOLD / no clearance", "HOLD"
    decoded = decode_action(action)
    if decoded is None:
        return "HOLD / no clearance", "HOLD"
    slot, choice = decoded
    aircraft = state.get("aircraft", [])
    if slot >= len(aircraft):
        return f"invalid slot {slot}", "OTHER"
    ac = aircraft[slot]
    label = aircraft_label(ac)
    if choice == CHOICE_SHORT:
        return f"RELEASE {label} via SHORT", "SHORT"
    if choice == CHOICE_BYPASS:
        return f"RELEASE {label} via BYPASS", "BYPASS"
    if choice == CHOICE_ADVANCE:
        nxt = ac.get("next_position")
        if nxt == RUNWAY_SERVICE:
            return f"RUNWAY service for {label}", "RUNWAY"
        if nxt == RUNWAY_THRESHOLD:
            return f"QUEUE {label} at threshold", "QUEUE"
        if nxt == APRON:
            return f"DONE / clear {label} to apron", "DONE"
        return f"TAXI {label} to {NODE_NAMES.get(nxt, 'next')}", "TAXI"
    return f"action {action}", "OTHER"


def run_baseline_rollout(name: str, policy_fn: Callable[[dict], int]) -> Rollout:
    sim = AirportSimulator(scenario=SCENARIO)
    state = sim.reset(seed=SEED)
    frames = [Frame(state, "START", "START", 0.0)]
    cumulative_reward = 0.0
    done = False

    while not done:
        state["_sim"] = sim
        action = int(policy_fn(state))
        state.pop("_sim", None)
        label, kind = describe_action(state, action)
        state, reward, done, _ = sim.step(action)
        cumulative_reward += float(reward)
        frames.append(Frame(state, label, kind, cumulative_reward))

    return Rollout(name, frames, episode_metrics(sim))


def run_ppo_rollout(name: str, checkpoint: str) -> Rollout:
    env = AirportEnv(
        scenario=SCENARIO,
        obs_mode="poisson_features",
        strategic_noop=True,
    )
    model = MaskablePPO.load(checkpoint)
    obs, _ = env.reset(seed=SEED)
    state = env._sim._get_state()
    frames = [Frame(state, "START", "START", 0.0)]
    cumulative_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        label, kind = describe_action(env._sim._get_state(), int(action))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        cumulative_reward += float(reward)
        frames.append(Frame(env._sim._get_state(), label, kind, cumulative_reward))

    metrics = episode_metrics(env._sim)
    env.close()
    return Rollout(name, frames, metrics)


def color_for_aircraft(ac: dict) -> str:
    if "arrival" in ac["route_key"]:
        return "#219653"
    if ac["route_key"].endswith("_bypass"):
        return "#8e44ad"
    return "#2f80ed"


def action_color(kind: str) -> str:
    return {
        "HOLD": "#f39c12",
        "SHORT": "#2f80ed",
        "BYPASS": "#8e44ad",
        "RUNWAY": "#2c3e50",
        "QUEUE": "#34495e",
        "TAXI": "#16a085",
        "DONE": "#27ae60",
    }.get(kind, "#7f8c8d")


def _active_aircraft_by_id(state: dict) -> dict[int, dict]:
    return {
        ac["global_id"]: ac
        for ac in state["aircraft"]
        if ac["active"] and not ac["done"] and ac["position"] in NODE_POS
    }


def visual_aircraft(prev_state: dict, state: dict, alpha: float) -> list[tuple[dict, float, float]]:
    prev_aircraft = _active_aircraft_by_id(prev_state)
    current_by_id = {
        ac["global_id"]: ac
        for ac in state["aircraft"]
        if ac["global_id"] >= 0
    }
    aircraft = []
    drawn_ids = set()
    for ac in _active_aircraft_by_id(state).values():
        x, y = NODE_POS[ac["position"]]
        prev = prev_aircraft.get(ac["global_id"])
        if prev is not None and prev["position"] in NODE_POS:
            px, py = NODE_POS[prev["position"]]
            x = px + (x - px) * alpha
            y = py + (y - py) * alpha
        aircraft.append((ac, x, y))
        drawn_ids.add(ac["global_id"])

    # Arrivals complete as soon as they enter the apron, so the simulator marks
    # them done in the post-step state. Keep them visible for this interpolated
    # frame; otherwise they appear to vanish at Central instead of traveling to
    # Apron.
    for global_id, prev in prev_aircraft.items():
        if global_id in drawn_ids:
            continue
        current = current_by_id.get(global_id)
        next_node = prev.get("next_position")
        if current is None or not current.get("done") or next_node not in NODE_POS:
            continue
        px, py = NODE_POS[prev["position"]]
        nx, ny = NODE_POS[next_node]
        x = px + (nx - px) * alpha
        y = py + (ny - py) * alpha
        ghost = dict(prev)
        ghost["position"] = next_node
        aircraft.append((ghost, x, y))
    return aircraft


def draw_graph(
    ax,
    frame: Frame,
    rollout: Rollout,
    upto_step: int,
    prev_frame: Frame | None = None,
    alpha: float = 1.0,
) -> None:
    state = frame.state
    prev_state = prev_frame.state if prev_frame is not None else state
    step = int(state["step"])
    ax.clear()
    ax.set_xlim(-0.55, 7.65)
    ax.set_ylim(-0.95, 3.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#fbfcfc")

    for src, dst in EDGES:
        x1, y1 = NODE_POS[src]
        x2, y2 = NODE_POS[dst]
        linestyle = "--" if src in {SPOT_A, SPOT_B, BYPASS_I} and dst in {BYPASS_I, BYPASS_LINK} else "-"
        ax.plot([x1, x2], [y1, y2], color="#abb2b9", linewidth=2.0, linestyle=linestyle, zorder=1)

    ax.text(
        4.15,
        -0.58,
        "longer bypass route",
        fontsize=6.2,
        color="#7d3c98",
        ha="center",
        va="center",
        zorder=1,
    )

    occupied = {
        ac["position"]
        for ac in state["aircraft"]
        if ac["active"] and not ac["done"] and ac["position"] in NODE_POS
    }
    for node, (x, y) in NODE_POS.items():
        if node in {INTERSECTION_2, BYPASS_I}:
            face = "#fadbd8" if node in occupied else "#fdebd0"
            edge = "#c0392b"
        elif node in {RUNWAY_THRESHOLD, RUNWAY_SERVICE}:
            face = "#d6dbdf" if node in occupied else "#f4f6f7"
            edge = "#2c3e50"
        elif node in {RUNWAY_EXIT, APRON}:
            face = "#d5f5e3"
            edge = "#239b56"
        elif node in {GATE_A, GATE_B, GATE_C, GATE_D}:
            face = "#d6eaf8"
            edge = "#2874a6"
        else:
            face = "#f2f3f4"
            edge = "#7f8c8d"
        patch = FancyBboxPatch(
            (x - 0.28, y - 0.16),
            0.56,
            0.32,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x, y, NODE_LABELS[node], ha="center", va="center", fontsize=6.5, zorder=3)

    nearby_counts: dict[tuple[int, int], int] = {}
    for ac, x, y in visual_aircraft(prev_state, state, alpha):
        bucket = (round(x * 10), round(y * 10))
        idx = nearby_counts.get(bucket, 0)
        nearby_counts[bucket] = idx + 1
        # Keep aircraft centered on the taxiway line during interpolation.
        # Only separate exact overlaps slightly, horizontally, so motion still
        # reads as traveling along the route instead of floating above it.
        dx = idx * 0.13
        dy = 0.0
        circle = Circle(
                (x + dx, y + dy),
                0.13,
                facecolor=color_for_aircraft(ac),
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
        ax.add_patch(circle)
        ax.text(
            x + dx,
            y + dy,
            aircraft_label(ac),
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    dep_backlog = int(state.get("departure_backlog", 0))
    arr_backlog = int(state.get("arrival_backlog", 0))
    completed = len(state.get("completed_records", []))
    generated = int(state.get("generated_departures", 0) + state.get("generated_arrivals", 0))
    runway_remaining = int(state.get("runway_time_remaining", 0))
    action_box_color = action_color(frame.action_kind)

    ax.text(
        0.0,
        1.07,
        f"{rollout.name}  |  t={step:03d}/200  |  {phase_name(step)}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#17202a",
        ha="left",
    )
    ax.text(
        0.0,
        1.00,
        (
            f"completed/generated {completed}/{generated}   "
            f"depQ {dep_backlog}  arrQ {arr_backlog}   "
            f"runwayQ {state.get('runway_queue_len', 0)}/{state.get('runway_queue_capacity', 1)}   "
            f"runway busy {runway_remaining}"
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        color="#2c3e50",
        ha="left",
    )
    ax.text(
        0.0,
        -0.08,
        frame.action_label,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        ha="left",
        va="top",
        bbox={"facecolor": action_box_color, "edgecolor": action_box_color, "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.58,
        -0.08,
        f"cumulative reward {frame.cumulative_reward:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        color="#2c3e50",
        ha="left",
        va="top",
    )


def draw_timeseries(ax, frame_idx: int, rollout: Rollout) -> None:
    ax.clear()
    ax.set_facecolor("#f8f9fa")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="white", linewidth=0.8)
    ax.tick_params(length=0, labelsize=7)

    frames = rollout.frames[: frame_idx + 1]
    steps = [f.state["step"] for f in frames]
    backlog = [f.state.get("backlog_count", 0) for f in frames]
    completed = [len(f.state.get("completed_records", [])) for f in frames]

    ax.axvspan(0, 20, color="#d6eaf8", alpha=0.55, linewidth=0)
    ax.axvspan(20, 85, color="#fadbd8", alpha=0.55, linewidth=0)
    ax.axvspan(85, 200, color="#d5f5e3", alpha=0.42, linewidth=0)
    ax.plot(steps, backlog, color="#c0392b", linewidth=2, label="backlog")
    ax.plot(steps, completed, color="#239b56", linewidth=2, label="completed")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, max(58, max(backlog + completed + [1]) + 3))
    ax.set_xlabel("timestep", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def _frame_indices(runway: Rollout, ppo: Rollout) -> list[int]:
    max_frames = max(len(runway.frames), len(ppo.frames))
    indices = list(range(0, max_frames, FRAME_STRIDE))
    if indices[-1] != max_frames - 1:
        indices.append(max_frames - 1)
    return indices


def _render_plan(frame_indices: list[int]) -> list[tuple[int, int, float]]:
    plan = []
    for idx, current_idx in enumerate(frame_indices):
        next_idx = frame_indices[idx + 1] if idx + 1 < len(frame_indices) else current_idx
        if next_idx == current_idx:
            plan.append((current_idx, current_idx, 1.0))
            continue
        for substep in range(INTERP_STEPS):
            alpha = substep / float(INTERP_STEPS)
            plan.append((current_idx, next_idx, alpha))
    plan.append((frame_indices[-1], frame_indices[-1], 1.0))
    return plan


def make_figure():
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.05], hspace=0.32, wspace=0.12)
    axes = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    )
    fig.suptitle(
        "Same-Seed Poisson Burst Rollout: RunwayAware vs PPO Features + Strategic Hold",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    return fig, axes


def draw_comparison_frame(
    fig,
    axes,
    runway: Rollout,
    ppo: Rollout,
    prev_idx: int,
    idx: int,
    alpha: float,
) -> None:
    ax_graph_left, ax_graph_right, ax_ts_left, ax_ts_right = axes
    left_idx = min(idx, len(runway.frames) - 1)
    right_idx = min(idx, len(ppo.frames) - 1)
    left_prev_idx = min(prev_idx, len(runway.frames) - 1)
    right_prev_idx = min(prev_idx, len(ppo.frames) - 1)
    draw_graph(
        ax_graph_left,
        runway.frames[left_idx],
        runway,
        left_idx,
        prev_frame=runway.frames[left_prev_idx],
        alpha=alpha,
    )
    draw_graph(
        ax_graph_right,
        ppo.frames[right_idx],
        ppo,
        right_idx,
        prev_frame=ppo.frames[right_prev_idx],
        alpha=alpha,
    )
    draw_timeseries(ax_ts_left, left_idx, runway)
    draw_timeseries(ax_ts_right, right_idx, ppo)
    fig.canvas.draw_idle()


def export_frames(runway: Rollout, ppo: Rollout, render_plan: list[tuple[int, int, float]]) -> str:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for filename in os.listdir(FRAMES_DIR):
        if filename.endswith(".png"):
            os.remove(os.path.join(FRAMES_DIR, filename))

    fig, axes = make_figure()
    for seq, (prev_idx, idx, alpha) in enumerate(render_plan):
        draw_comparison_frame(fig, axes, runway, ppo, prev_idx, idx, alpha)
        out_path = os.path.join(FRAMES_DIR, f"frame_{seq:03d}_t{idx:03d}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FRAMES_DIR


def make_animation(runway: Rollout, ppo: Rollout, render_plan: list[tuple[int, int, float]]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = make_figure()

    def update(i: int):
        prev_idx, idx, alpha = render_plan[i]
        draw_comparison_frame(fig, axes, runway, ppo, prev_idx, idx, alpha)
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(render_plan),
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )

    out_path = os.path.join(OUT_DIR, "poisson_burst_runwayaware_vs_ppo_hold.gif")
    writer = animation.PillowWriter(fps=FPS)
    ani.save(out_path, writer=writer, dpi=105)
    plt.close(fig)
    return out_path


def main() -> None:
    runway = run_baseline_rollout("RunwayAware", runway_aware_policy)
    ppo = run_ppo_rollout("PPO features + hold", PPO_HOLD_CHECKPOINT)
    indices = _frame_indices(runway, ppo)
    plan = _render_plan(indices)
    frames_dir = export_frames(runway, ppo, plan)
    out_path = make_animation(runway, ppo, plan)
    print(f"saved: {out_path}")
    print(f"saved frames: {frames_dir} ({len(plan)} PNGs)")
    for rollout in [runway, ppo]:
        metrics = rollout.final_metrics
        print(
            f"{rollout.name}: reward={rollout.frames[-1].cumulative_reward:.1f}, "
            f"completed={metrics['completed_total']:.0f}/{metrics['generated_total']:.0f}, "
            f"censored_delay={metrics['mean_demand_delay_including_unserved']:.1f}, "
            f"unserved={metrics['unserved_total']:.0f}"
        )


if __name__ == "__main__":
    main()
