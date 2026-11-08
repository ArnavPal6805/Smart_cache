import argparse

from smartcache.core.cache import CacheSimulator
from smartcache.environment.workload import generate_workload
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_policy import DQNPolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/dqn_best.pt")
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    requests = generate_workload(
        number_of_requests=args.requests, seed=args.seed, drift=True
    )
    simulator = CacheSimulator(
        capacity=args.capacity, frequency_window=100, recency_window=1000
    )
    policies = [
        LRUPolicy(),
        LFUPolicy(),
        CostAwarePolicy(),
        DQNPolicy(args.checkpoint, capacity=args.capacity, device=args.device),
    ]
    for policy in policies:
        result = simulator.run(requests, policy)
        print(
            f"{policy.name:<10} "
            f"hit_ratio={result['hit_ratio']:.3f} "
            f"miss_cost_ms={result['total_miss_cost_ms']:.1f} "
            f"net_cost_ms={result['estimated_net_cost_ms']:.3f} "
            f"evictions={result['evictions']}"
        )


if __name__ == "__main__":
    main()
