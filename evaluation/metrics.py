# Each episode result dict has keys:
#   total_reward, steps, conflicts, illegal_moves, completions, timed_out


def mean_reward(results: list[dict]) -> float:
    # TODO: return average of result["total_reward"] across all episodes.
    pass


def mean_taxi_time(results: list[dict]) -> float:
    # TODO: return average of result["steps"] across all episodes. Lower is better.
    pass


def conflict_rate(results: list[dict]) -> float:
    # TODO: return average of result["conflicts"] across all episodes.
    pass


def mean_throughput(results: list[dict]) -> float:
    # TODO: return average of result["completions"] across all episodes. Higher is better.
    pass


def timeout_rate(results: list[dict]) -> float:
    # TODO: return fraction of episodes where result["timed_out"] is True.
    pass


def summary_table(results_by_policy: dict[str, list[dict]]) -> str:
    # TODO: build and return a formatted comparison table string.
    #
    # For each policy name in results_by_policy, compute all 5 metrics above
    # and format into a table with columns:
    #   Policy | Mean Reward | Mean Steps | Conflicts/ep | Throughput | Timeout%
    #
    # Example output:
    #   Policy           | Mean Reward | Mean Steps | Conflicts/ep | Throughput | Timeout%
    #   -----------------|-------------|------------|--------------|------------|--------
    #   PPO              |   -42.3     |    14.1    |     0.12     |    2.88    |  0.0%
    #   FCFS             |   -61.2     |    19.4    |     0.31     |    2.69    |  2.0%
    #   ConflictAware    |   -48.7     |    15.8    |     0.08     |    2.92    |  0.0%
    pass
