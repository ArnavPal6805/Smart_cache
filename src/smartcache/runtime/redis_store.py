from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import redis


class RedisSharedCache:
    """Shared payload and eviction metadata storage backed by Redis."""

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        namespace: str = "smartcache:byte:v2",
        capacity_bytes: int = 24_576,
        max_objects: int = 16,
        lock_timeout_seconds: int = 30,
    ) -> None:
        if capacity_bytes <= 0 or max_objects <= 0:
            raise ValueError("Cache limits must be positive")
        self.client = redis.Redis.from_url(url, decode_responses=False)
        self.namespace = namespace.rstrip(":")
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.lock_timeout_seconds = lock_timeout_seconds
        self.meta_key = f"{self.namespace}:metadata"
        self.bytes_key = f"{self.namespace}:used_bytes"
        self.lock_key = f"{self.namespace}:admission_lock"
        self.payload_prefix = f"{self.namespace}:payload:"

    def _payload_key(self, key: str) -> str:
        return f"{self.payload_prefix}{key}"

    def get(self, key: str) -> bytes | None:
        payload = self.client.get(self._payload_key(key))
        return bytes(payload) if payload is not None else None

    def metadata(self, key: str) -> dict[str, Any] | None:
        value = self.client.hget(self.meta_key, key)
        if value is None:
            return None
        return json.loads(value)

    def keys(self) -> list[str]:
        return [key.decode() if isinstance(key, bytes) else key for key in self.client.hkeys(self.meta_key)]

    @contextmanager
    def transaction_lock(self) -> Iterator[None]:
        lock = self.client.lock(
            self.lock_key,
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.lock_timeout_seconds,
        )
        acquired = lock.acquire()
        if not acquired:
            raise TimeoutError("Timed out acquiring Redis cache admission lock")
        try:
            yield
        finally:
            lock.release()

    def admit(
        self,
        key: str,
        payload: bytes,
        retrieval_cost_ms: float,
        evicted_keys: list[str],
        position: int,
    ) -> None:
        size = len(payload)
        if size > self.capacity_bytes:
            return
        for victim in evicted_keys:
            self.remove(victim)
        metadata = {
            "key": key,
            "size_bytes": size,
            "retrieval_cost_ms": float(retrieval_cost_ms),
            "last_access": int(position),
            "frequency": 1,
        }
        self.client.set(self._payload_key(key), payload)
        self.client.hset(self.meta_key, key, json.dumps(metadata))
        self.client.incrby(self.bytes_key, size)

    def touch(self, key: str, position: int) -> None:
        metadata = self.metadata(key)
        if metadata is None:
            return
        metadata["last_access"] = int(position)
        metadata["frequency"] = int(metadata.get("frequency", 0)) + 1
        self.client.hset(self.meta_key, key, json.dumps(metadata))

    def remove(self, key: str) -> None:
        metadata = self.metadata(key)
        pipe = self.client.pipeline()
        pipe.delete(self._payload_key(key))
        pipe.hdel(self.meta_key, key)
        if metadata is not None:
            pipe.decrby(self.bytes_key, int(metadata["size_bytes"]))
        pipe.execute()

    def stats(self) -> dict[str, int]:
        used = int(self.client.get(self.bytes_key) or 0)
        return {
            "used_bytes": max(0, used),
            "object_count": self.client.hlen(self.meta_key),
        }

    def reconcile(self) -> dict[str, int]:
        """Repair byte counters and remove metadata without payloads."""
        with self.transaction_lock():
            used = 0
            orphaned = 0
            for key in self.keys():
                metadata = self.metadata(key)
                if metadata is None or self.get(key) is None:
                    self.client.hdel(self.meta_key, key)
                    self.client.delete(self._payload_key(key))
                    orphaned += 1
                    continue
                used += len(self.get(key) or b"")
            self.client.set(self.bytes_key, used)
            return {
                "used_bytes": used,
                "object_count": self.client.hlen(self.meta_key),
                "orphaned_entries": orphaned,
            }
