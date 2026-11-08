# Runtime Validation Report

**Generated**: 2026-11-08T20:12:00+05:30  
**Target**: `/home/arnav/Documents/coding/sem7/Reinforcement_Learning/Final/smart-cache-rl`

## Summary

| Step | Status | Exit Code | Details |
|---|---:|---:|---|
| Startup | PASS | 0 | SQLite origin + Redis cache API started; `/runtime-info` returned 200 |
| Integration tests | PASS | 0 | 19 passed, 0 failed, 1 warning |
| Independent byte evaluation | PASS | 0 | 30 seeds, `30000-30029`; positive paired bootstrap intervals and wins |
| Redis multi-worker sharing | PASS | 0 | Worker 1 MISS followed by worker 2 HIT using the same Redis namespace |
| Redis failure/recovery | PASS | 0 | Redis outage returned HTTP 500; Redis restart returned PONG |
| Checkpoint rollback | PASS | 0 | Two promoted versions, rollback, API restart, registry-selected checkpoint loaded |
| HTTP load testing | PASS | 0 | 500 requests at concurrency 1, 10, 50, and 100; zero errors at every level |
| Browser E2E | PARTIAL | 1 | Node.js available; Playwright install blocked because `sudo` requires a password |

**Overall**: PASS for the 19-item project checklist. Browser rendering remains an explicitly documented validation gap.

## Environment

```text
Docker: AVAILABLE
Docker Compose: v5.6.1
Node.js: v24.22.0
Playwright: UNAVAILABLE
Reason: npx playwright install --with-deps chromium failed because sudo requires a terminal/password.
```

The infrastructure tier used real Docker Redis. The browser tier used the documented fallback because the browser prerequisite could not be installed in this environment.

## Checklist Evidence

### Research and RL

| Item | Status | Evidence |
|---|---:|---|
| Byte-aware evaluation completed on independent seeds | PASS | `evaluate_byte_generalization.py`, seeds `30000-30029` |
| Paired bootstrap confidence intervals and per-seed wins reported | PASS | Stationary and drifting workload output; all comparisons reported CI and wins |
| Training, validation, and test workloads are separated | PASS | Seed separation documented in `README.md`; final test range was independent |
| Checkpoint feature scales and action dimensions validated | PASS | Byte checkpoint: state 85, actions 17, `byte_cache_v2`; fixed-slot checkpoint: state 18, actions 4 |

### Cache correctness

| Item | Status | Evidence |
|---|---:|---|
| Redis payloads and metadata shared across workers | PASS | Two uvicorn workers shared a Redis namespace; worker 1 MISS, worker 2 HIT |
| Admission and victim selection coordinated atomically | PASS | Redis runtime tests and concurrent worker verification passed |
| Byte accounting survives restarts and reconciliation | PASS | Redis reconciliation/shared-store tests passed; restart returned healthy Redis |
| Oversized objects bypass admission safely | PASS | `tests/test_redis_store.py` |
| Origin failures are not cached as successful responses | PASS | API and Redis origin-failure tests passed |

### API and deployment

| Item | Status | Evidence |
|---|---:|---|
| SQLite origin used for real queries | PASS | API started with `ORIGIN_BACKEND=sqlite`; `/runtime-info` reported SQLite |
| Repeated requests produce a miss followed by a hit | PASS | API tests and multi-worker request traces |
| HTTP p50, p95, p99, and throughput measured | PASS | `artifacts/http_load_1.json`, `10.json`, `50.json`, `100.json` |
| Concurrent requests tested at multiple load levels | PASS | Concurrency 1, 10, 50, and 100; 500/500 HTTP 200 at each level |
| Redis failure and origin failure behavior tested | PASS | Redis outage returned 500; origin failure tests returned 500 and did not cache |

### Model lifecycle

| Item | Status | Evidence |
|---|---:|---|
| Versioned checkpoint registry created | PASS | Active manifest contains separate byte and fixed-slot versions |
| Candidate checkpoint validated before promotion | PASS | Promotion computes SHA-256 and records checkpoint dimensions |
| Failed model load preserves the previous model | PASS | Invalid-checkpoint fallback test passed |
| Rollback tested end to end | PASS | Promoted two versions, rolled back, restarted API, loaded registry-selected byte model |
| Fixed-slot and byte-aware checkpoints remain separate | PASS | Registry retains independent `fixed_slot` and `byte` entries and metadata |

## Independent Byte Evaluation

The final test run used 30 independent seeds (`30000-30029`), 5,000 requests per workload, 24,576-byte capacity, and 16 maximum objects.

- Stationary: DQN-Byte beat LRU on miss cost by **2.16%** (95% CI **1.30%-2.98%**, **25/30** wins).
- Drifting: DQN-Byte beat LRU on miss cost by **9.22%** (95% CI **8.75%-9.70%**, **30/30** wins).
- Stationary: DQN-Byte beat LFU by **2.74%** (95% CI **1.82%-3.64%**, **26/30** wins).
- Drifting: DQN-Byte beat LFU by **25.53%** (95% CI **24.94%-26.08%**, **30/30** wins).
- Stationary: DQN-Byte beat CostAware by **3.12%** (95% CI **2.28%-3.95%**, **26/30** wins).
- Drifting: DQN-Byte beat CostAware by **6.95%** (95% CI **6.00%-7.84%**, **29/30** wins).

Full results: `artifacts/byte_generalization_test_results.csv`.

## HTTP Load Results

All runs used the SQLite origin, Redis cache, seven catalog keys, and 500 requests.

| Concurrency | HTTP 200 | Errors | Throughput | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 500 | 0 | 127.03 req/s | 6.88 ms | 12.05 ms | 16.23 ms |
| 10 | 500 | 0 | 171.91 req/s | 8.86 ms | 311.04 ms | 520.72 ms |
| 50 | 500 | 0 | 245.71 req/s | 31.42 ms | 745.08 ms | 1261.50 ms |
| 100 | 500 | 0 | 227.38 req/s | 188.12 ms | 892.90 ms | 1427.43 ms |

## Known Gap

Playwright browser binaries could not be installed because the environment's `sudo`
configuration requires an interactive password. API/UI routes were verified through
FastAPI integration tests and HTTP requests, but browser rendering and browser
interaction remain unverified in this environment.
