BASE_CONFIG = {
    # TODO: fill in values for the standard training run
    # "n_steps":         2048,    # rollout buffer size (steps per update)
    # "batch_size":      64,      # minibatch size for gradient updates
    # "n_epochs":        10,      # passes over collected data per update
    # "gamma":           0.99,    # discount factor
    # "learning_rate":   3e-4,
    # "clip_range":      0.2,     # PPO clip epsilon
    # "ent_coef":        0.01,    # entropy bonus coefficient (encourages exploration)
    # "total_timesteps": 500_000,
}

FAST_CONFIG = {
    # TODO: copy BASE_CONFIG but set total_timesteps to 50_000
    # Use this for quick sanity checks during development.
}

LONG_CONFIG = {
    # TODO: copy BASE_CONFIG but set total_timesteps to 1_000_000
    # Use this for final training runs before evaluation.
}
