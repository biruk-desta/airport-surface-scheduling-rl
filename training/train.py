import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.airport_env import AirportEnv
from training.config import BASE_CONFIG


def make_env(scenario: str = "default"):
      # Vectorised env for faster rollout collection
      return make_vec_env(lambda: AirportEnv(scenario=scenario), n_envs=4)


def build_model(env):
    # Remove total_timesteps from PPO constructor kwargs
    ppo_kwargs = {k: v for k, v in BASE_CONFIG.items() if k != "total_timesteps"}
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./experiments/tb_logs/",
        **ppo_kwargs,
    )


def train(model, total_timesteps: int):
    model.learn(total_timesteps=total_timesteps)
    return model
    # SB3 will print a progress table every n_steps * n_envs steps.


def evaluate(model, scenario: str = "default", n_episodes: int = 5):

    env = AirportEnv(scenario=scenario)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc

        rewards.append(total_r)
        print(f"  Episode {ep+1}: reward={total_r:.1f}")

    env.close()
    mean_reward = sum(rewards) / n_episodes
    print(f"Mean reward over {n_episodes} episodes: {mean_reward:.2f}")


def save(model, path: str = "experiments/ppo_airport"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}.zip")


if __name__ == "__main__":
      env   = make_env()
      model = build_model(env)
      model = train(model, BASE_CONFIG["total_timesteps"])
      evaluate(model)
      save(model)

    # To view training curves:
    #   source rl/bin/activate
    #   tensorboard --logdir experiments/tb_logs/
    #   open http://localhost:6006
