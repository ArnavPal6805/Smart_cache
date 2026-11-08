from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import HTMLResponse

from smartcache.api.demo_origin import ORIGIN_SPECS, OriginSpec, load_origin_payload
from smartcache.api.sqlite_origin import SQLiteOrigin
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_byte_policy import DQNBytePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.runtime.cache_service import ByteCacheRuntime
from smartcache.api.simulation import build_simulation


def create_app(
    *,
    policy_factory: Callable[[], object] | None = None,
    origin_specs: Mapping[str, OriginSpec] = ORIGIN_SPECS,
    origin_loader: Callable[[str], bytes] = load_origin_payload,
    capacity_bytes: int | None = None,
    max_objects: int | None = None,
    frequency_window: int | None = None,
    recency_window: int | None = None,
    cost_scale_ms: float | None = None,
    size_scale_bytes: float | None = None,
) -> FastAPI:
    use_sqlite = os.getenv("ORIGIN_BACKEND", "demo").lower() == "sqlite"
    sqlite_origin = (
        SQLiteOrigin(os.getenv("SQLITE_DATABASE", "data/origin.sqlite"))
        if use_sqlite
        else None
    )
    effective_origin_specs = (
        sqlite_origin.specs_from_profile(os.getenv("ORIGIN_PROFILE"))
        if sqlite_origin is not None
        else origin_specs
    )
    effective_origin_loader = (
        sqlite_origin.loader() if sqlite_origin is not None else origin_loader
    )
    settings = {
        "capacity_bytes": capacity_bytes or int(os.getenv("CACHE_CAPACITY_BYTES", "24576")),
        "max_objects": max_objects or int(os.getenv("CACHE_MAX_OBJECTS", "16")),
        "frequency_window": frequency_window or 100,
        "recency_window": recency_window or 1000,
        "cost_scale_ms": cost_scale_ms or 250.0,
        "size_scale_bytes": size_scale_bytes or 32768.0,
    }

    def build_policy():
        if policy_factory is not None:
            return policy_factory()
        policy_name = os.getenv("CACHE_POLICY", "dqn-byte").lower()
        if policy_name == "lru":
            return LRUPolicy()
        if policy_name == "lfu":
            return LFUPolicy()
        if policy_name == "cost-aware":
            return CostAwarePolicy()
        if policy_name == "dqn-byte":
            return DQNBytePolicy(
                checkpoint_path=os.getenv(
                    "CACHE_CHECKPOINT", "artifacts/byte_dqn_best.pt"
                ),
                capacity_bytes=settings["capacity_bytes"],
                max_objects=settings["max_objects"],
                frequency_window=settings["frequency_window"],
                recency_window=settings["recency_window"],
                device=os.getenv("CACHE_DEVICE", "cpu"),
            )
        raise ValueError(f"Unknown CACHE_POLICY: {policy_name}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cache = ByteCacheRuntime(
            policy=build_policy(),
            origin_specs=effective_origin_specs,
            origin_loader=effective_origin_loader,
            **settings,
        )
        yield

    app = FastAPI(title="SmartCache RL", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartCache RL</title>
<style>
body {{ margin:0; background:#0e1624; color:#eef4ff; font:16px system-ui,sans-serif; }}
main {{ max-width:900px; margin:auto; padding:42px 24px; }}
.card {{ background:#172235; border:1px solid #2b3b55; border-radius:14px; padding:22px; margin:18px 0; }}
a,button {{ color:#fff; background:#1d4f87; border:1px solid #3479c7; border-radius:8px; padding:10px 14px; text-decoration:none; margin:4px; display:inline-block; }}
.muted {{ color:#9dafc8; }} code {{ color:#9bd1ff; }}
</style></head><body><main>
<h1>SmartCache RL</h1>
<p class="muted">Cost-aware cache eviction with live API diagnostics.</p>
<div class="card"><h2>Current runtime</h2>
<p>Origin: <strong>{'SQLite' if use_sqlite else 'Demo simulator'}</strong></p>
<p>Policy: <strong>{os.getenv("CACHE_POLICY", "dqn-byte")}</strong></p>
<p>Cache: <strong>{settings["capacity_bytes"]:,} bytes</strong>, up to
<strong>{settings["max_objects"]}</strong> objects</p></div>
<div class="card"><h2>Open</h2>
<a href="/explorer">Simulation Explorer</a>
<a href="/docs">API documentation</a>
<a href="/objects">Object catalog</a>
<a href="/metrics">Live metrics</a></div>
<div class="card"><h2>Try a request</h2>
<p class="muted">Requesting the same object twice demonstrates MISS then HIT.</p>
<code>curl -i /objects/aggregation</code>
</div></main></body></html>"""

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/objects")
    def objects():
        return {"objects": sorted(effective_origin_specs)}

    @app.get("/runtime-info")
    def runtime_info():
        return {
            "origin_backend": "sqlite" if use_sqlite else "demo",
            "policy": os.getenv("CACHE_POLICY", "dqn-byte"),
            "capacity_bytes": settings["capacity_bytes"],
            "max_objects": settings["max_objects"],
            "objects": sorted(effective_origin_specs),
        }

    @app.get("/objects/{key}")
    def get_object(key: str):
        if key not in effective_origin_specs:
            raise HTTPException(status_code=404, detail=f"Unknown object: {key}")
        result = app.state.cache.get(key)
        headers = {
            "X-Cache": "HIT" if result.hit else "MISS",
            "X-Backend-Latency-Ms": f"{result.backend_latency_ms:.3f}",
            "X-Policy-Time-Ms": f"{result.policy_time_ms:.3f}",
            "X-Cache-Used-Bytes": str(result.used_bytes),
            "X-Cache-Bypassed": str(result.bypassed).lower(),
        }
        return Response(content=result.payload, media_type="application/octet-stream", headers=headers)

    @app.get("/metrics")
    def metrics():
        return app.state.cache.snapshot_metrics()

    @app.get("/simulation")
    def simulation(
        scenario: str = "stationary",
        seed: int = 2026,
        requests: int = 30,
        capacity: int = 3,
    ):
        checkpoint = os.getenv("CACHE_CHECKPOINT", "artifacts/byte_dqn_best.pt")
        try:
            return build_simulation(
                scenario=scenario,
                seed=seed,
                requests_count=requests,
                capacity=capacity,
                checkpoint=checkpoint,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/explorer", response_class=HTMLResponse)
    def explorer():
        html_path = os.path.join(
            os.path.dirname(__file__),
            "simulation.html",
        )
        with open(html_path, encoding="utf-8") as file:
            return file.read()

    return app


app = create_app()
