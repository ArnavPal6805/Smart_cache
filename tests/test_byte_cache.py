from smartcache.core.byte_cache import ByteCacheSimulator
from smartcache.policies.lru import LRUPolicy


def test_multiple_evictions_make_room():
    requests = [
        {"key": "a", "cost": 1.0, "size": 40},
        {"key": "b", "cost": 2.0, "size": 40},
        {"key": "c", "cost": 3.0, "size": 70},
    ]
    result = ByteCacheSimulator(capacity_bytes=100).run(requests, LRUPolicy())
    assert result["misses"] == 3
    assert result["evictions"] == 2
    assert result["peak_bytes"] <= 100
    assert result["final_bytes"] == 70
    assert result["final_cache"] == ["c"]
    assert result["total_miss_cost_ms"] == 6.0


def test_oversized_object_does_not_evict_cached_object():
    requests = [
        {"key": "small", "cost": 2.0, "size": 20},
        {"key": "oversized", "cost": 80.0, "size": 101},
        {"key": "small", "cost": 2.0, "size": 20},
    ]
    result = ByteCacheSimulator(capacity_bytes=100).run(requests, LRUPolicy())
    assert result["hits"] == 1
    assert result["evictions"] == 0
    assert result["final_cache"] == ["small"]
    assert result["final_bytes"] == 20
