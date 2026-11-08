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


def test_home_and_runtime_info_are_available():
    app = create_app(
        policy_factory=LRUPolicy,
        origin_specs={"alpha": OriginSpec(2.0, 10, ord("A"))},
        origin_loader=lambda key: b"A" * 10,
        capacity_bytes=100,
        max_objects=4,
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        info = client.get("/runtime-info")
        assert info.status_code == 200
        assert info.json()["origin_backend"] == "demo"
        assert info.json()["objects"] == ["alpha"]
        assert client.get("/catalog").status_code == 200
        assert client.get("/metrics-ui").status_code == 200


def test_origin_failure_is_returned_and_not_cached():
    specs = {"broken": OriginSpec(2.0, 10, ord("B"))}
    app = create_app(
        policy_factory=LRUPolicy,
        origin_specs=specs,
        origin_loader=lambda key: (_ for _ in ()).throw(
            OSError("origin unavailable")
        ),
        capacity_bytes=100,
        max_objects=4,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/objects/broken")
        assert response.status_code == 500
        assert app.state.cache.snapshot_metrics()["objects"] == 0
