import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import AirportSimulator, NODE_NAMES, INTERSECTION, GATE_NODES, RUNWAY, APRON

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 2D layout positions for each node ID
NODE_POS = {
    0: (0.0, 1.0),   # Gate A
    1: (0.0, 0.0),   # Gate B
    2: (1.5, 1.0),   # Entry A
    3: (1.5, 0.0),   # Entry B
    4: (3.0, 0.5),   # Intersection
    5: (4.5, 0.5),   # Runway
    6: (4.5, -0.8),  # Runway Exit
    7: (3.0, -0.8),  # Arr Taxi
    8: (0.0, -0.8),  # Apron
}

EDGES = [
    (0, 2), (1, 3),
    (2, 4), (3, 4),
    (4, 5),
    (6, 7), (7, 4),
    (4, 8),
]

AIRCRAFT_COLORS = ["#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]


def _node_color(node_id: int, intersection_occupied: bool) -> str:
    if node_id == INTERSECTION:
        return "#e74c3c" if intersection_occupied else "#2ecc71"
    if node_id in GATE_NODES:
        return "#3498db"
    if node_id in (RUNWAY, APRON):
        return "#95a5a6"
    return "#dfe6e9"


def render_frame(state: dict, title: str = "", ax=None, show: bool = True):
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(-1.5, 1.7)
    ax.axis("off")
    ax.set_title(f"Step {state['step']}  {title}", fontsize=11)

    # Draw edges
    for src, dst in EDGES:
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#bdc3c7", lw=1.5,
                            shrinkA=14, shrinkB=14),
        )

    # Draw nodes
    int_occ = state["intersection_occupied"]
    for node_id, (x, y) in NODE_POS.items():
        color = _node_color(node_id, int_occ)
        circle = plt.Circle((x, y), 0.18, color=color, zorder=3, ec="#2c3e50", lw=1.2)
        ax.add_patch(circle)
        label = NODE_NAMES.get(node_id, str(node_id)).replace("_", "\n")
        ax.text(x, y - 0.34, label, ha="center", va="top", fontsize=6.5, color="#2c3e50")

    # Draw active aircraft
    active_aircraft = [ac for ac in state["aircraft"] if ac["active"]]
    offsets = [(-0.08, 0.08), (0.08, 0.08), (-0.08, -0.08), (0.08, -0.08)]
    for ac in active_aircraft:
        pos = ac["position"]
        if pos not in NODE_POS:
            continue
        x, y = NODE_POS[pos]
        dx, dy = offsets[ac["id"] % 4]
        color = AIRCRAFT_COLORS[ac["id"] % len(AIRCRAFT_COLORS)]
        ax.plot(x + dx, y + dy, "^", color=color, markersize=12, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(x + dx, y + dy + 0.22, f"AC{ac['id']}", ha="center",
                fontsize=7, color=color, fontweight="bold", zorder=6)

    # Legend
    patches = [mpatches.Patch(color=AIRCRAFT_COLORS[ac["id"]], label=f"AC{ac['id']} ({ac['route_key']})")
               for ac in active_aircraft]
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)

    if show and created:
        plt.tight_layout()
        plt.show()


def render_episode(sim: AirportSimulator, policy_fn, pause: float = 0.5):
    state = sim.reset()
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.ion()

    done = False
    while not done:
        ax.clear()
        render_frame(state, ax=ax, show=False)
        plt.tight_layout()
        plt.pause(pause)
        action = policy_fn(state)
        state, _, done, _ = sim.step(action)

    # Final frame
    ax.clear()
    render_frame(state, title="Episode Complete", ax=ax, show=False)
    plt.tight_layout()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    sim = AirportSimulator(scenario="default")
    state = sim.reset()
    render_frame(state, title="Initial State")
