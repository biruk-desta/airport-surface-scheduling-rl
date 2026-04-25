import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.airport_env import AirportEnv
from experiments import MODELS, CHECKPOINT_DIR

N_ENVS     = 4
N_EVAL_EPS = 5
TB_LOG_DIR = "experiments/tb_logs/"


def checkpoint_path(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, name)


def already_trained(name: str) -> bool:
    p = checkpoint_path(name)
    return os.path.exists(p) or os.path.exists(f"{p}.zip")


def train_model(name: str, cfg: dict) -> None:
    scenario = cfg["scenario"]
    config   = cfg["config"]

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"  scenario={scenario}  timesteps={config['total_timesteps']:,}")
    print(f"{'='*60}")

    env = make_vec_env(lambda: AirportEnv(scenario=scenario), n_envs=N_ENVS)
    ppo_kwargs = {k: v for k, v in config.items() if k != "total_timesteps"}
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log=TB_LOG_DIR, **ppo_kwargs)
    model.learn(total_timesteps=config["total_timesteps"])

    # Quick eval
    eval_env = AirportEnv(scenario=scenario)
    rewards = []
    for _ in range(N_EVAL_EPS):
        obs, _ = eval_env.reset()
        done, total_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
    eval_env.close()
    print(f"  Quick eval ({N_EVAL_EPS} eps): mean reward = {sum(rewards)/N_EVAL_EPS:.1f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(checkpoint_path(name))
    print(f"  Saved: {checkpoint_path(name)}.zip")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain all models even if checkpoints exist")
    parser.add_argument("--only", nargs="+", metavar="MODEL",
                        help="Train only these model names")
    args = parser.parse_args()

    to_train = args.only if args.only else list(MODELS.keys())

    for name in to_train:
        if name not in MODELS:
            print(f"  [skip] unknown model '{name}'")
            continue
        if not args.retrain and already_trained(name):
            print(f"  [skip] {name} — checkpoint exists (use --retrain to force)")
            continue
        train_model(name, MODELS[name])

    print("\n  Training complete. Run evaluation/evaluate.py next.")
