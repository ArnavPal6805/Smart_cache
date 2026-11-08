from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT = Path("artifacts/generalization_results.csv")
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 12345


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def summarize_reduction(
    differences: list[float],
    baseline_costs: list[float],
    dqn_costs: list[float],
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    reduction_percentages = np.asarray(
        [
            100.0 * difference / baseline
            for difference, baseline in zip(differences, baseline_costs)
            if baseline > 0
        ],
        dtype=np.float64,
    )
    if len(reduction_percentages) < 2:
        raise ValueError("At least two valid paired seeds are required")
    interval = bootstrap_interval(reduction_percentages, rng)
    win_rate = float(np.mean(np.asarray(dqn_costs) < np.asarray(baseline_costs)))
    return (
        float(np.mean(reduction_percentages)),
        float(interval[0]),
        float(interval[1]),
        win_rate,
    )


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Results file not found: {INPUT}")

    with INPUT.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    if not rows:
        raise ValueError("The results CSV is empty")

    required = {"scenario", "seed", "policy"}
    missing = required - set(columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {columns}"
        )

    raw_column = (
        "total_miss_cost_ms"
        if "total_miss_cost_ms" in columns
        else "miss_cost_ms"
        if "miss_cost_ms" in columns
        else None
    )
    if raw_column is None or "net_cost_ms" not in columns:
        raise ValueError(
            "Could not find miss-cost and net-cost columns. "
            f"Available columns: {columns}"
        )

    grouped: defaultdict[tuple[str, str], dict[str, dict[str, float]]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (row["scenario"], row["seed"])
        grouped[key][row["policy"]] = {
            "raw": float(row[raw_column]),
            "net": float(row["net_cost_ms"]),
        }

    scenarios = sorted({scenario for scenario, _ in grouped})
    baselines = sorted(
        {
            policy
            for policies in grouped.values()
            for policy in policies
            if policy != "DQN"
        }
    )
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Input: {INPUT}")
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")
    print("Intervals: paired bootstrap 95% confidence intervals")

    for scenario in scenarios:
        print(f"\n{scenario.capitalize()} workloads")
        for baseline in baselines:
            pairs = [
                policies
                for (current_scenario, _), policies in grouped.items()
                if current_scenario == scenario
                and "DQN" in policies
                and baseline in policies
            ]
            if len(pairs) < 2:
                print(f"  DQN vs {baseline}: insufficient paired observations")
                continue

            dqn_raw = [pair["DQN"]["raw"] for pair in pairs]
            baseline_raw = [pair[baseline]["raw"] for pair in pairs]
            dqn_net = [pair["DQN"]["net"] for pair in pairs]
            baseline_net = [pair[baseline]["net"] for pair in pairs]

            raw_stats = summarize_reduction(
                [baseline - dqn for baseline, dqn in zip(baseline_raw, dqn_raw)],
                baseline_raw,
                dqn_raw,
                rng,
            )
            net_stats = summarize_reduction(
                [baseline - dqn for baseline, dqn in zip(baseline_net, dqn_net)],
                baseline_net,
                dqn_net,
                rng,
            )
            print(
                f"  DQN vs {baseline}: "
                f"miss-cost reduction={raw_stats[0]:.2f}% "
                f"(95% CI {raw_stats[1]:.2f}% to {raw_stats[2]:.2f}%), "
                f"wins={raw_stats[3] * 100:.1f}% "
                f"| net-cost reduction={net_stats[0]:.2f}% "
                f"(95% CI {net_stats[1]:.2f}% to {net_stats[2]:.2f}%), "
                f"wins={net_stats[3] * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
