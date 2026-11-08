import argparse

from smartcache.core.byte_cache import ByteCacheSimulator
from smartcache.environment.workload import generate_workload
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_byte_policy import DQNBytePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy


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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    requests = generate_workload(
        number_of_requests=args.requests,
        seed=args.seed,
        drift=True,
    )
    simulator = ByteCacheSimulator(
        capacity_bytes=args.capacity_bytes,
        max_objects=args.max_objects,
        frequency_window=args.frequency_window,
        recency_window=args.recency_window,
        cost_scale_ms=args.cost_scale_ms,
        size_scale_bytes=args.size_scale_bytes,
    )
    policies = [
        LRUPolicy(),
        LFUPolicy(),
        CostAwarePolicy(recency_tau=20.0),
        DQNBytePolicy(
            checkpoint_path=args.checkpoint,
            capacity_bytes=args.capacity_bytes,
            max_objects=args.max_objects,
            frequency_window=args.frequency_window,
            recency_window=args.recency_window,
            device=args.device,
        ),
    ]

    print(f"Requests: {args.requests}")
    print(f"Capacity bytes: {args.capacity_bytes}")
    print(f"Maximum objects: {args.max_objects}")
    for policy in policies:
        result = simulator.run(requests, policy)
        print(
            f"{policy.name:<10} "
            f"hit_ratio={result['hit_ratio']:.3f} "
            f"miss_cost_ms={result['total_miss_cost_ms']:.1f} "
            f"net_cost_ms={result['estimated_net_cost_ms']:.3f} "
            f"evictions={result['evictions']} "
            f"peak_bytes={result['peak_bytes']} "
            f"final_bytes={result['final_bytes']}"
        )


if __name__ == "__main__":
    main()
