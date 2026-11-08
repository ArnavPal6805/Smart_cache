from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter

import httpx


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--keys", default="metadata,user-profile,aggregation,embedding,report")
    parser.add_argument("--output", default="artifacts/http_load.json")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("requests and concurrency must be positive")
    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    if not keys:
        parser.error("at least one key is required")
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    cache_results: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def request(client: httpx.AsyncClient, index: int) -> None:
        key = keys[index % len(keys)]
        async with semaphore:
            request_started = time.perf_counter()
            try:
                response = await client.get(f"/objects/{key}")
                latencies.append((time.perf_counter() - request_started) * 1000)
                statuses[str(response.status_code)] += 1
                cache_results[response.headers.get("x-cache", "MISSING")] += 1
            except httpx.HTTPError as error:
                errors[type(error).__name__] += 1

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        await asyncio.gather(*(request(client, index) for index in range(args.requests)))
    elapsed = time.perf_counter() - started
    completed = sum(statuses.values())
    report = {
        "requests": args.requests,
        "completed": completed,
        "errors": sum(errors.values()),
        "error_rate": (sum(errors.values()) / args.requests),
        "concurrency": args.concurrency,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": completed / elapsed if elapsed else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "status_counts": dict(statuses),
        "cache_results": dict(cache_results),
        "errors_by_type": dict(errors),
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
