# Each episode result dict has keys:
#   total_reward, steps, conflicts, illegal_moves, completions, timed_out


def mean_reward(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["total_reward"] for result in results) / len(results)


def mean_episode_steps(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["steps"] for result in results) / len(results)


def conflict_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["conflicts"] for result in results) / len(results)


def illegal_action_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["illegal_moves"] for result in results) / len(results)


def noop_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("noop_count", 0.0) for result in results) / len(results)


def noop_when_legal_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("noop_when_legal_count", 0.0) for result in results) / len(results)


def mean_throughput(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["completions"] for result in results) / len(results)


def mean_delay(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("mean_delay", 0.0) for result in results) / len(results)


def mean_demand_delay_including_unserved(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(
        result.get("mean_demand_delay_including_unserved", result.get("mean_delay", 0.0))
        for result in results
    ) / len(results)


def mean_completed_demand_delay(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(
        result.get("completed_mean_demand_delay", result.get("mean_delay", 0.0))
        for result in results
    ) / len(results)


def mean_unserved_total(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("unserved_total", 0.0) for result in results) / len(results)


def mean_runway_utilization(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("runway_utilization", 0.0) for result in results) / len(results)


def mean_arrival_delay(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("arrival_delay", 0.0) for result in results) / len(results)


def mean_departure_delay(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("departure_delay", 0.0) for result in results) / len(results)


def mean_short_routes(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("short_routes", 0.0) for result in results) / len(results)


def mean_bypass_routes(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("bypass_routes", 0.0) for result in results) / len(results)


def mean_generated(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("generated_total", 0.0) for result in results) / len(results)


def mean_completed_total(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result.get("completed_total", 0.0) for result in results) / len(results)


def mean_final_backlog(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(
        result.get("departure_backlog", 0.0) + result.get("arrival_backlog", 0.0)
        for result in results
    ) / len(results)


def mean_peak_backlog(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(
        result.get("max_departure_backlog", 0.0) + result.get("max_arrival_backlog", 0.0)
        for result in results
    ) / len(results)


def timeout_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result["timed_out"]) / len(results)


def summary_table(results_by_policy: dict[str, list[dict]]) -> str:
    headers = [
        "Policy",
        "N",
        "Mean Reward",
        "Mean Steps",
        "SurfaceDelay",
        "CompletedDemand",
        "InclUnserved",
        "ArrDelay",
        "DepDelay",
        "Runway%",
        "S/B",
        "Conflicts/ep",
        "Illegal/ep",
        "NoopLegal/ep",
        "Throughput",
        "Demand",
        "Unserved",
        "PeakQ",
        "FinalQ",
        "Timeout%",
    ]

    rows = []
    for policy, results in results_by_policy.items():
        rows.append([
            policy,
            str(len(results)),
            f"{mean_reward(results):.1f}",
            f"{mean_episode_steps(results):.1f}",
            f"{mean_delay(results):.1f}",
            f"{mean_completed_demand_delay(results):.1f}",
            f"{mean_demand_delay_including_unserved(results):.1f}",
            f"{mean_arrival_delay(results):.1f}",
            f"{mean_departure_delay(results):.1f}",
            f"{mean_runway_utilization(results) * 100:.1f}%",
            f"{mean_short_routes(results):.1f}/{mean_bypass_routes(results):.1f}",
            f"{conflict_rate(results):.2f}",
            f"{illegal_action_rate(results):.2f}",
            f"{noop_when_legal_rate(results):.2f}",
            f"{mean_throughput(results):.2f}",
            f"{mean_completed_total(results):.1f}/{mean_generated(results):.1f}",
            f"{mean_unserved_total(results):.1f}",
            f"{mean_peak_backlog(results):.1f}",
            f"{mean_final_backlog(results):.1f}",
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
