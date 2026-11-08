from smartcache.core.cache import CacheSimulator
from smartcache.policies.lru import LRUPolicy


def test_lru_hit_miss_and_eviction():
    requests = [
        {"key": "a", "cost": 2.0, "size": 100},
        {"key": "b", "cost": 10.0, "size": 100},
        {"key": "a", "cost": 2.0, "size": 100},
        {"key": "c", "cost": 20.0, "size": 100},
        {"key": "a", "cost": 2.0, "size": 100},
    ]
    result = CacheSimulator(capacity=2).run(requests, LRUPolicy())
    assert result["requests"] == 5
    assert result["hits"] == 2
    assert result["misses"] == 3
    assert result["evictions"] == 1
    assert result["total_miss_cost_ms"] == 32.0
    assert result["final_cache"] == ["a", "c"]


def test_request_trace_does_not_mutate():
    requests = [
        {"key": "a", "cost": 2.0, "size": 100},
        {"key": "b", "cost": 10.0, "size": 100},
    ]
    original = [dict(request) for request in requests]
    CacheSimulator(capacity=1).run(requests, LRUPolicy())
    assert requests == original
