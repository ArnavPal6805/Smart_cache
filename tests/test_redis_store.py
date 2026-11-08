import os
import uuid

import pytest

from smartcache.runtime.redis_store import RedisSharedCache
from smartcache.runtime.redis_cache_service import RedisByteCacheRuntime
from smartcache.api.demo_origin import OriginSpec
from smartcache.policies.lru import LRUPolicy


@pytest.fixture
def cache():
    instance = RedisSharedCache(
        url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/15"),
        namespace=f"test:{uuid.uuid4().hex}",
        capacity_bytes=1024,
        max_objects=4,
    )
    try:
        instance.client.ping()
    except Exception as error:
        pytest.skip(f"Redis unavailable: {error}")
    yield instance
    for key in instance.client.scan_iter(match=f"{instance.namespace}:*"):
        instance.client.delete(key)


def test_reconcile_repairs_byte_counter(cache):
    with cache.transaction_lock():
        cache.admit("a", b"a" * 100, 10.0, [], 1)
        cache.client.set(cache.bytes_key, 9999)
    result = cache.reconcile()
    assert result["used_bytes"] == 100
    assert result["object_count"] == 1
    assert cache.stats()["used_bytes"] == 100


def test_two_runtime_instances_share_payload_and_metadata(cache):
    specs = {"alpha": OriginSpec(2.0, 5, ord("A"))}
    calls = []

    def loader(key):
        calls.append(key)
        return b"alpha"

    first = RedisByteCacheRuntime(cache, LRUPolicy(), specs, loader, capacity_bytes=100)
    second = RedisByteCacheRuntime(cache, LRUPolicy(), specs, loader, capacity_bytes=100)

    assert first.get("alpha").hit is False
    result = second.get("alpha")

    assert result.hit is True
    assert calls == ["alpha"]
    assert cache.stats() == {"used_bytes": 5, "object_count": 1}


def test_oversized_objects_bypass_without_metadata(cache):
    specs = {"large": OriginSpec(10.0, 2000, ord("L"))}
    runtime = RedisByteCacheRuntime(
        cache, LRUPolicy(), specs, lambda key: b"L" * 2000, capacity_bytes=100
    )

    result = runtime.get("large")

    assert result.bypassed is True
    assert cache.keys() == []
    assert cache.stats()["used_bytes"] == 0


def test_origin_failure_is_not_cached(cache):
    specs = {"broken": OriginSpec(10.0, 5, ord("B"))}

    def loader(key):
        raise OSError("origin unavailable")

    runtime = RedisByteCacheRuntime(cache, LRUPolicy(), specs, loader)

    with pytest.raises(OSError):
        runtime.get("broken")

    assert cache.keys() == []
    assert cache.stats()["used_bytes"] == 0
