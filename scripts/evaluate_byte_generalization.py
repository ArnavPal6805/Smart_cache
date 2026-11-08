from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from smartcache.core.byte_cache import ByteCacheSimulator
from smartcache.environment.workload import generate_workload
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_byte_policy import DQNBytePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy


def bootstrap_ci(values, rng, samples=20_000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two independent seeds are required")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/byte_dqn_best.pt")
    parser.add_argument("--capacity-bytes", type=int, default=24576)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--frequency-window", type=int, default=100)
    parser.add_argument("--recency-window", type=int, default=1000)
    parser.add_argument("--cost-scale-ms", type=float, default=250.0)
    parser.add_argument("--size-scale-bytes", type=float, default=32768.0)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="artifacts/byte_generalization_results.csv"
    )
    args = parser.parse_args()
    if args.seed_count < 2:
        parser.error("Use at least two seeds")

    policies = [
        LRUPolicy(),
        LFUPolicy(),
        CostAwarePolicy(recency_tau=20.0),
        DQNBytePolicy(
            args.checkpoint,
            capacity_bytes=args.capacity_bytes,
            max_objects=args.max_objects,
            frequency_window=args.frequency_window,
            recency_window=args.recency_window,
            device=args.device,
        ),
    ]
    simulator = ByteCacheSimulator(
        capacity_bytes=args.capacity_bytes,
        max_objects=args.max_objects,
        frequency_window=args.frequency_window,
        recency_window=args.recency_window,
        cost_scale_ms=args.cost_scale_ms,
        size_scale_bytes=args.size_scale_bytes,
    )
    rows = []
    for scenario, drift in (("stationary", False), ("drifting", True)):
        for offset in range(args.seed_count):
            seed = args.seed_start + offset
            requests = generate_workload(args.requests, seed=seed, drift=drift)
            for policy in policies:
                result = simulator.run(requests, policy)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": policy.name,
                        "total_miss_cost_ms": result["total_miss_cost_ms"],
                        "net_cost_ms": result["estimated_net_cost_ms"],
                        "hit_ratio": result["hit_ratio"],
                        "evictions": result["evictions"],
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(12345)
    for scenario, _ in (("stationary", False), ("drifting", True)):
        print(f"{scenario.capitalize()} workloads")
        for baseline in ("LRU", "LFU", "CostAware"):
            reductions = []
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                values = {
                    row["policy"]: float(row["total_miss_cost_ms"])
                    for row in rows
                    if row["scenario"] == scenario and row["seed"] == seed
                }
                reductions.append(
                    100.0 * (values[baseline] - values["DQN-Byte"]) / values[baseline]
                )
            mean_value, lower, upper = bootstrap_ci(reductions, rng)
            print(
                f"  DQN-Byte vs {baseline}: "
                f"{mean_value:.2f}% (95% CI {lower:.2f}% to {upper:.2f}%)"
            )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
