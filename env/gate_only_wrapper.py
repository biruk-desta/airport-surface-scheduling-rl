import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from baselines.conflict_aware import conflict_aware_policy


class GateOnlyWrapper(gym.Wrapper):
    """
    Ablation wrapper for Experiment 3.

    Restricts the agent to gate-release decisions only. If the agent picks
    an action that targets an aircraft already taxiing (past its gate), the
    action is overridden by the ConflictAware heuristic so intersection
    timing is handled automatically.

    This isolates the contribution of joint gate+intersection control:
    compare a model trained with this wrapper (gate-only) against the full
    PPO agent (joint control) to see whether controlling the intersection
    on top of gate releases adds value.
    """

    def step(self, action):
        action = int(action)
        state = self.env._sim._get_state()

        if action > 0:
            target_idx = action - 1
            aircraft = state["aircraft"]
            if target_idx < len(aircraft):
                ac = aircraft[target_idx]
                # Agent tried to advance a taxiing aircraft — override with heuristic
                if ac["active"] and not ac["at_gate"]:
                    action = conflict_aware_policy(state)

        return self.env.step(action)
