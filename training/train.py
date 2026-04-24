import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.airport_env import AirportEnv
from env.gate_only_wrapper import GateOnlyWrapper
from training.config import BASE_CONFIG


def make_env(scenario: str = "default"):
    return make_vec_env(lambda: AirportEnv(scenario=scenario), n_envs=4)


def make_gate_only_env(scenario: str = "default"):
    return make_vec_env(lambda: GateOnlyWrapper(AirportEnv(scenario=scenario)), n_envs=4)


def build_model(env):
    ppo_kwargs = {k: v for k, v in BASE_CONFIG.items() if k != "total_timesteps"}
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        tensorboard_log="./experiments/tb_logs/",
        **ppo_kwargs,
    )


def train(model, total_timesteps: int):
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, scenario: str = "default", n_episodes: int = 5,
             gate_only: bool = False):
    if gate_only:
        env = GateOnlyWrapper(AirportEnv(scenario=scenario))
    else:
        env = AirportEnv(scenario=scenario)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, total_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        print(f"  Episode {ep+1}: reward={total_r:.1f}")

    env.close()
    print(f"  Mean reward: {sum(rewards)/n_episodes:.2f}")


def save(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"  Saved: {path}.zip")


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    ts = BASE_CONFIG["total_timesteps"]

    # --- Model 1: full joint control on default (main model) ---
    print("\n" + "="*55)
    print("  Model 1: PPO full joint control — default scenario")
    print("="*55)
    env = make_env("default")
    model = build_model(env)
    model = train(model, ts)
    evaluate(model, "default")
    save(model, "experiments/ppo_airport")

    # --- Model 2: full joint control on heavy (generalization test) ---
    print("\n" + "="*55)
    print("  Model 2: PPO full joint control — heavy scenario")
    print("="*55)
    env = make_env("heavy")
    model = build_model(env)
    model = train(model, ts)
    evaluate(model, "heavy")
    save(model, "experiments/ppo_airport_heavy")

    # --- Model 3: gate-only on default (Experiment 3 ablation) ---
    # Gate-only is a simpler task (agent only learns gate timing), so lower
    # entropy is sufficient and converges more reliably.
    print("\n" + "="*55)
    print("  Model 3: PPO gate-only control — default scenario")
    print("="*55)
    env = make_gate_only_env("default")
    gate_only_kwargs = {k: v for k, v in BASE_CONFIG.items() if k != "total_timesteps"}
    gate_only_kwargs["ent_coef"] = 0.01
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./experiments/tb_logs/", **gate_only_kwargs)
    model = train(model, ts)
    evaluate(model, "default", gate_only=True)
    save(model, "experiments/ppo_gate_only")

    print("\n✓ All models trained. Run evaluation/evaluate.py next.")

    # To view training curves:
    #   source rl/bin/activate
    #   tensorboard --logdir experiments/tb_logs/
    #   open http://localhost:6006
