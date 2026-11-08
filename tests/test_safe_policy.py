from smartcache.core.interfaces import CacheEntry
from smartcache.policies.safe_policy import SafePolicy


class FailingPolicy:
    name = "failing"

    def choose_victim(self, cache, incoming, timestamp, context=None):
        raise ValueError("inference failed")


class FallbackPolicy:
    name = "fallback"

    def choose_victim(self, cache, incoming, timestamp, context=None):
        return cache[0].key


def test_safe_policy_uses_fallback_and_counts_failure():
    policy = SafePolicy(FailingPolicy(), FallbackPolicy())
    cache = [CacheEntry("alpha", 1.0, 10, 1, 1)]

    assert policy.choose_victim(cache, {"key": "beta"}, 2) == "alpha"
    assert policy.fallback_errors == 1
