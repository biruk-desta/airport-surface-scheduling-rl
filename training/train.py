import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.airport_env import AirportEnv
from training.config import BASE_CONFIG


def make_env(scenario: str = "default"):
    # TODO: return a vectorised env with n_envs=4 for faster rollout collection.
    # Use make_vec_env with a lambda so each worker gets its own AirportEnv instance.
    # Example:
    #   return make_vec_env(lambda: AirportEnv(scenario=scenario), n_envs=4)
    pass


def build_model(env):
    # TODO: instantiate and return a PPO model with BASE_CONFIG hyperparameters.
    # Pass tensorboard_log="./experiments/tb_logs/" so training curves are logged.
    # Strip "total_timesteps" from the config before passing to PPO constructor.
    # Example:
    #   ppo_kwargs = {k: v for k, v in BASE_CONFIG.items() if k != "total_timesteps"}
    #   return PPO("MlpPolicy", env, verbose=1,
    #              tensorboard_log="./experiments/tb_logs/", **ppo_kwargs)
    pass


def train(model, total_timesteps: int):
    # TODO: call model.learn(total_timesteps=total_timesteps) to run training.
    # SB3 will print a progress table every n_steps * n_envs steps.
    # Returns the trained model.
    pass


def evaluate(model, scenario: str = "default", n_episodes: int = 5):
    # TODO: run n_episodes with the trained model and print reward per episode.
    # Use a fresh AirportEnv (not vectorised) for clean single-episode rollouts.
    # Example loop:
    #   env = AirportEnv(scenario=scenario)
    #   for ep in range(n_episodes):
    #       obs, _ = env.reset()
    #       done, total_r = False, 0.0
    #       while not done:
    #           action, _ = model.predict(obs, deterministic=True)
    #           obs, r, term, trunc, _ = env.step(action)
    #           total_r += r; done = term or trunc
    #       print(f"  Episode {ep+1}: reward={total_r:.1f}")
    pass


def save(model, path: str = "experiments/ppo_airport"):
    # TODO: create the experiments/ directory if needed, then call model.save(path).
    # SB3 automatically appends .zip to the path.
    pass


if __name__ == "__main__":
    # TODO: wire everything together:
    #   env   = make_env()
    #   model = build_model(env)
    #   model = train(model, BASE_CONFIG["total_timesteps"])
    #   evaluate(model)
    #   save(model)

    # To view training curves:
    #   source rl/bin/activate
    #   tensorboard --logdir experiments/tb_logs/
    #   open http://localhost:6006
    pass
