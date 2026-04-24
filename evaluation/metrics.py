# Each episode result dict has keys:
#   total_reward, steps, conflicts, illegal_moves, completions, timed_out


def mean_reward(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["total_reward"] for result in results) / len(results)


def mean_taxi_time(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["steps"] for result in results) / len(results)


def conflict_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["conflicts"] for result in results) / len(results)


def mean_throughput(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["completions"] for result in results) / len(results)


def timeout_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result["timed_out"]) / len(results)


def summary_table(results_by_policy: dict[str, list[dict]]) -> str:
    headers = [
        "Policy",
        "Mean Reward",
        "Mean Steps",
        "Conflicts/ep",
        "Throughput",
        "Timeout%",
    ]

    rows = []
    for policy, results in results_by_policy.items():
        rows.append([
            policy,
            f"{mean_reward(results):.1f}",
            f"{mean_taxi_time(results):.1f}",
            f"{conflict_rate(results):.2f}",
            f"{mean_throughput(results):.2f}",
            f"{timeout_rate(results) * 100:.1f}%",
        ])

    widths = [
        max([len(headers[col])] + [len(row[col]) for row in rows])
        for col in range(len(headers))
    ]

    def _format_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(width) if idx == 0 else value.rjust(width)
            for idx, (value, width) in enumerate(zip(values, widths))
        )

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)
