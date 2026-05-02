"""Training entry point for the airport-surface benchmark.

RL algorithms are imported from Stable-Baselines3 and SB3-Contrib:
PPO is provided by ``stable_baselines3`` and MaskablePPO is provided by
``sb3_contrib``. The simulator, Gymnasium wrapper, action masks, scenarios,
baselines, metrics, and evaluation scripts are project-specific code.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
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


def train_model(name: str, cfg: dict, seed: int = 0, maskable: bool = False) -> None:
    scenario = cfg["scenario"]
    config   = cfg["config"]
    enhanced_obs = bool(cfg.get("enhanced_obs", False))
    obs_mode = cfg.get("obs_mode")
    strategic_noop = bool(cfg.get("strategic_noop", False))
    save_name = f"{name}_maskable_seed{seed}" if maskable else f"{name}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(
        f"  scenario={scenario}  timesteps={config['total_timesteps']:,}  "
        f"seed={seed}  algorithm={'MaskablePPO' if maskable else 'PPO'}"
    )
    print(f"{'='*60}")

    env = make_vec_env(
        lambda: Monitor(AirportEnv(
            scenario=scenario,
            enhanced_obs=enhanced_obs,
            obs_mode=obs_mode,
            strategic_noop=strategic_noop,
        )),
        n_envs=N_ENVS,
        seed=seed,
    )
    ppo_kwargs = {k: v for k, v in config.items() if k != "total_timesteps"}
    model_cls = MaskablePPO if maskable else PPO
    model = model_cls("MlpPolicy", env, verbose=1, seed=seed,
                      tensorboard_log=TB_LOG_DIR, **ppo_kwargs)
    model.learn(total_timesteps=config["total_timesteps"])

    # Quick eval
    eval_env = AirportEnv(
        scenario=scenario,
        enhanced_obs=enhanced_obs,
        obs_mode=obs_mode,
        strategic_noop=strategic_noop,
    )
    rewards = []
    for _ in range(N_EVAL_EPS):
        obs, _ = eval_env.reset(seed=seed)
        done, total_r = False, 0.0
        while not done:
            if maskable:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=eval_env.action_masks(),
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
    eval_env.close()
    print(f"  Quick eval ({N_EVAL_EPS} eps): mean reward = {sum(rewards)/N_EVAL_EPS:.1f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(checkpoint_path(save_name))
    print(f"  Saved: {checkpoint_path(save_name)}.zip")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain all models even if checkpoints exist")
    parser.add_argument("--only", nargs="+", metavar="MODEL",
                        help="Train only these model names")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed")
    parser.add_argument("--maskable", action="store_true",
                        help="Train MaskablePPO using AirportEnv.action_masks()")
    args = parser.parse_args()

    to_train = args.only if args.only else list(MODELS.keys())

    for name in to_train:
        if name not in MODELS:
            print(f"  [skip] unknown model '{name}'")
            continue
        save_name = f"{name}_maskable_seed{args.seed}" if args.maskable else f"{name}_seed{args.seed}"
        if not args.retrain and already_trained(save_name):
            print(f"  [skip] {save_name} — checkpoint exists (use --retrain to force)")
            continue
        train_model(name, MODELS[name], seed=args.seed, maskable=args.maskable)

    print("\n  Training complete. Run evaluation/evaluate.py next.")
