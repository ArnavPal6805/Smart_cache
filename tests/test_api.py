from fastapi.testclient import TestClient

from smartcache.api.app import create_app
from smartcache.api.demo_origin import OriginSpec
from smartcache.policies.lru import LRUPolicy


def test_first_request_misses_and_second_hits():
    specs = {"alpha": OriginSpec(2.0, 10, ord("A"))}
    load_calls = []

    def loader(key):
        load_calls.append(key)
        return b"A" * 10

    app = create_app(
        policy_factory=LRUPolicy,
        origin_specs=specs,
        origin_loader=loader,
        capacity_bytes=100,
        max_objects=4,
    )
    with TestClient(app) as client:
        first = client.get("/objects/alpha")
        second = client.get("/objects/alpha")
        assert first.status_code == 200
        assert first.headers["X-Cache"] == "MISS"
        assert len(first.content) == 10
        assert second.status_code == 200
        assert second.headers["X-Cache"] == "HIT"
        assert load_calls == ["alpha"]
