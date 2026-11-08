import os
import uuid

import pytest

from smartcache.runtime.redis_store import RedisSharedCache


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
