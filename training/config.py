BASE_CONFIG = {
    "n_steps":         2048,    # rollout buffer size (steps per update)
    "batch_size":      64,      # minibatch size for gradient updates
    "n_epochs":        10,      # passes over collected data per update
    "gamma":           0.99,    # discount factor
    "learning_rate":   3e-4,
    "clip_range":      0.2,     # PPO clip epsilon
    "ent_coef":        0.05,    # entropy bonus coefficient (encourages exploration)
    "total_timesteps": 500_000,
}

FAST_CONFIG = {
    **BASE_CONFIG,
    "total_timesteps": 50_000,  # for quick testing and debugging
}

LONG_CONFIG = {
    **BASE_CONFIG,
    "total_timesteps": 1_000_000,  # for final training runs before evaluation
}
