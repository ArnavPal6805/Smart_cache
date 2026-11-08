from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from smartcache.core.cache import CacheSimulator
from smartcache.environment.workload import generate_workload
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_policy import DQNPolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy


def format_summary(values: list[float]) -> str:
    if len(values) < 2:
        return f"{mean(values):.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cache policies across independent workloads"
    )
    parser.add_argument("--checkpoint", default="artifacts/dqn_best.pt")
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="artifacts/generalization_results.csv",
    )
    args = parser.parse_args()

    if args.capacity < 1 or args.requests < 1:
        parser.error("Capacity and request count must be positive")
    if args.seed_count < 2:
        parser.error("Use at least two seeds")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    simulator = CacheSimulator(
        capacity=args.capacity,
        frequency_window=100,
        recency_window=1000,
    )
    policies = [
        LRUPolicy(),
        LFUPolicy(),
        CostAwarePolicy(recency_tau=20.0),
        DQNPolicy(
            checkpoint_path=checkpoint,
            capacity=args.capacity,
            device=args.device,
        ),
    ]
    policy_names = [policy.name for policy in policies]
    records: list[dict[str, object]] = []
    scenarios = [("stationary", False), ("drifting", True)]

    print(f"Cache capacity: {args.capacity}")
    print(f"Requests per trace: {args.requests}")
    print(f"Independent seeds: {args.seed_count}")
    print(f"Policies: {', '.join(policy_names)}")
    print()

    for scenario, drift in scenarios:
        print(f"Evaluating {scenario} workloads...")
        for offset in range(args.seed_count):
            seed = args.seed_start + offset
            requests = generate_workload(
                number_of_requests=args.requests,
                seed=seed,
                drift=drift,
            )
            results = {
                policy.name: simulator.run(requests, policy)
                for policy in policies
            }
            dqn_cost = float(results["DQN"]["total_miss_cost_ms"])
            dqn_net_cost = float(results["DQN"]["estimated_net_cost_ms"])
            for policy_name, result in results.items():
                miss_cost = float(result["total_miss_cost_ms"])
                net_cost = float(result["estimated_net_cost_ms"])
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": policy_name,
                        "requests": result["requests"],
                        "hits": result["hits"],
                        "misses": result["misses"],
                        "hit_ratio": result["hit_ratio"],
                        "evictions": result["evictions"],
                        "total_miss_cost_ms": miss_cost,
                        "policy_time_ms": result["policy_time_ms"],
                        "net_cost_ms": net_cost,
                        "dqn_cost_reduction_ms": (
                            miss_cost - dqn_cost if policy_name != "DQN" else 0.0
                        ),
                        "dqn_net_cost_reduction_ms": (
                            net_cost - dqn_net_cost if policy_name != "DQN" else 0.0
                        ),
                    }
                )
            print(f"  seed={seed} complete")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print()
    print(f"Saved results to {output}")
    for scenario, _ in scenarios:
        print(f"\n{scenario.capitalize()} summary:")
        for policy_name in policy_names:
            values = [
                float(record["total_miss_cost_ms"])
                for record in records
                if record["scenario"] == scenario
                and record["policy"] == policy_name
            ]
            net_values = [
                float(record["net_cost_ms"])
                for record in records
                if record["scenario"] == scenario
                and record["policy"] == policy_name
            ]
            print(
                f"  {policy_name:<10} "
                f"miss_cost_ms={format_summary(values)} "
                f"net_cost_ms={format_summary(net_values)}"
            )

        for baseline in policy_names:
            if baseline == "DQN":
                continue
            reductions = [
                float(record["dqn_cost_reduction_ms"])
                for record in records
                if record["scenario"] == scenario
                and record["policy"] == baseline
            ]
            net_reductions = [
                float(record["dqn_net_cost_reduction_ms"])
                for record in records
                if record["scenario"] == scenario
                and record["policy"] == baseline
            ]
            print(
                f"  DQN vs {baseline:<10} "
                f"miss_cost_reduction_ms={format_summary(reductions)} "
                f"net_cost_reduction_ms={format_summary(net_reductions)}"
            )


if __name__ == "__main__":
    main()
