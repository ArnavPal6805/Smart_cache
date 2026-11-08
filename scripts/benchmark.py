import random

from smartcache.core.cache import CacheSimulator
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy


def generate_workload(
    number_of_requests: int = 2000,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    objects = [
        {"key": "metadata", "cost": 2.0, "size": 512},
        {"key": "user-profile", "cost": 4.0, "size": 2048},
        {"key": "api-response", "cost": 15.0, "size": 4096},
        {"key": "aggregation", "cost": 180.0, "size": 8192},
        {"key": "embedding", "cost": 250.0, "size": 16384},
        {"key": "search-index", "cost": 90.0, "size": 32768},
        {"key": "report", "cost": 120.0, "size": 4096},
    ]
    probabilities = [0.45, 0.18, 0.13, 0.10, 0.08, 0.04, 0.02]
    selected = rng.choices(objects, weights=probabilities, k=number_of_requests)
    return [dict(request) for request in selected]


if __name__ == "__main__":
    requests = generate_workload()
    simulator = CacheSimulator(capacity=3)
    for policy in (LRUPolicy(), LFUPolicy(), CostAwarePolicy()):
        result = simulator.run(requests, policy)
        print(
            f"{policy.name:<10} "
            f"hit_ratio={result['hit_ratio']:.3f} "
            f"miss_cost_ms={result['total_miss_cost_ms']:.1f} "
            f"net_cost_ms={result['estimated_net_cost_ms']:.3f} "
            f"evictions={result['evictions']}"
        )
