# SmartCache RL

Cost-aware reinforcement learning for cache eviction.

SmartCache RL compares classical cache replacement policies with lightweight
Deep Q-Networks (DQN). Unlike hit-rate-only policies, the learning reward
penalizes a miss by the requested object's retrieval cost:

```text
reward = 0 on a hit
reward = -retrieval_cost_ms on a miss
```

The project contains two experiment tracks:

1. **Fixed-slot cache**: a request occupies one of a fixed number of slots.
2. **Byte-capacity cache**: objects have real sizes, and multiple victims may
   be evicted to admit one request.

It also includes a FastAPI service and a browser-based Simulation Explorer for
watching identical request streams replayed through DQN, LRU, LFU, and
CostAware policies.

## Status

Implemented:

- Fixed-slot and byte-capacity cache simulators
- LRU, LFU, CostAware, fixed-slot DQN, and byte-aware DQN policies
- Cost-weighted rewards and action masking
- Double DQN targets with experience replay
- Stationary and distribution-drift workloads
- Multi-seed generalization evaluation with paired bootstrap intervals
- Thread-safe runtime cache service
- FastAPI HTTP application
- Interactive simulation explorer
- SQLite-backed origin with measured query latency and response sizes
- Browser landing page, object catalog, and auto-refreshing live metrics page
- Redis-backed shared cache storage with coordinated admission
- HTTP load-test reporting p50/p95/p99 latency and throughput
- Versioned checkpoint registry with SHA-256 promotion and rollback
- Docker image definition

The byte-aware DQN is a separate model from the fixed-slot DQN. Their state
and action dimensions are different, so checkpoints must not be mixed.

The existing byte-aware model remains `byte_cache_v2`. A history-aware
`byte_cache_v3` model is intentionally not created without a separate training
and evaluation run.

## Repository layout

```text
smart-cache-rl/
├── artifacts/
│   ├── dqn_best.pt                    # fixed-slot best checkpoint
│   ├── dqn_latest.pt                  # fixed-slot latest checkpoint
│   ├── dqn_pre_mask_fix.pt            # fixed-slot comparison checkpoint
│   ├── byte_dqn_best.pt               # byte-aware best checkpoint
│   ├── byte_dqn_latest.pt             # byte-aware latest checkpoint
│   ├── generalization_results.csv     # fixed-slot multi-seed results
│   └── byte_generalization_results.csv # byte-capacity multi-seed results
├── scripts/
│   ├── init_demo_db.py                 # SQLite origin database generator
│   ├── profile_sqlite_origin.py        # measured SQL latency/size profile
│   ├── test_environment.py             # fixed-slot smoke test
│   ├── benchmark.py                    # fixed-slot baseline benchmark
│   ├── benchmark_rl.py                 # fixed-slot DQN benchmark
│   ├── train_dqn.py                   # fixed-slot DQN training
│   ├── evaluate_generalization.py     # fixed-slot multi-seed evaluation
│   ├── analyze_generalization.py      # paired bootstrap analysis
│   ├── benchmark_byte.py              # byte-capacity baseline benchmark
│   ├── train_byte_dqn.py              # byte-aware DQN training
│   ├── benchmark_byte_rl.py           # byte-aware DQN benchmark
│   ├── evaluate_byte_generalization.py # byte-aware multi-seed evaluation
│   ├── load_test_http.py              # concurrent HTTP latency benchmark
│   └── checkpoint_registry.py         # checkpoint promotion/rollback
├── src/smartcache/
│   ├── api/
│   │   ├── app.py                     # FastAPI application
│   │   ├── demo_origin.py             # deterministic simulated backend
│   │   ├── sqlite_origin.py           # read-only SQLite origin adapter
│   │   ├── simulation.py              # replay/comparison API
│   │   └── simulation.html            # browser Simulation Explorer
│   ├── core/
│   │   ├── interfaces.py              # CacheEntry and policy protocol
│   │   ├── cache.py                   # fixed-slot simulator
│   │   └── byte_cache.py              # byte-capacity simulator
│   ├── environment/
│   │   ├── cache_env.py               # fixed-slot Gymnasium environment
│   │   ├── state_encoder.py           # fixed-slot state encoder
│   │   ├── byte_cache_env.py          # multi-step byte-aware environment
│   │   ├── byte_state_encoder.py      # byte-aware state encoder
│   │   └── workload.py                # stationary/drifting workloads
│   ├── policies/
│   │   ├── lru.py                     # least recently used
│   │   ├── lfu.py                     # least frequently used
│   │   ├── cost_aware.py              # static cost-aware heuristic
│   │   ├── dqn_policy.py              # fixed-slot DQN adapter
│   │   └── dqn_byte_policy.py         # byte-aware DQN adapter
│   ├── runtime/
│   │   ├── cache_service.py           # thread-safe in-memory cache
│   │   ├── redis_store.py             # shared Redis payload/metadata store
│   │   └── redis_cache_service.py     # Redis-backed cache runtime
│   └── training/
│       ├── dqn_agent.py               # Q-network, Double DQN, checkpoints
│       └── replay_buffer.py           # masked transition replay
├── tests/
│   ├── test_cache_simulator.py
│   ├── test_byte_cache.py
│   ├── test_byte_cache_env.py
│   ├── test_sqlite_origin.py
│   ├── test_redis_store.py
│   ├── test_checkpoint_registry.py
│   ├── test_safe_policy.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── pyproject.toml
```

## Requirements and installation

Python 3.11 or newer is required. A virtual environment is recommended.

```bash
cd smart-cache-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package dependencies include NumPy, PyTorch, Gymnasium, pandas, PyYAML,
FastAPI, Uvicorn, Redis, pytest, and HTTPX.

## Real SQLite origin

The default API still uses the deterministic simulated origin. To create a
real local SQLite origin and measure actual SQL execution time and serialized
response sizes, run:

```bash
python scripts/init_demo_db.py
python scripts/profile_sqlite_origin.py \
    --database data/origin.sqlite \
    --repeats 15
```

The database contains deterministic synthetic products and purchases. The
adapter executes real read-only SQL for metadata, user profiles, API rows,
aggregations, embeddings, reports, and a search index. This is a local
integration demonstrat[github.com/ArnavPal6805/Smart_cache](https://github.com/ArnavPal6805/Smart_cache)ion, not a production database benchmark.

Use the SQLite origin with the API:

```bash
export ORIGIN_BACKEND=sqlite
export SQLITE_DATABASE=data/origin.sqlite
export ORIGIN_PROFILE=configs/sqlite_profile.json
export CACHE_POLICY=lru

uvicorn smartcache.api.app:app \
    --host 127.0.0.1 \
    --port 8000
```

The profile supplies the DQN/cache cost and size estimates while every miss
still measures the actual SQLite load duration.

Generated local files are ignored by Git:

```text
data/origin.sqlite
configs/sqlite_profile.json
```

## Run the tests and smoke checks

Run the complete test suite:

```bash
pytest -q
```

Run the fixed-slot environment smoke test:

```bash
python scripts/test_environment.py
```

The fixed-slot environment uses `capacity + 1` actions:

- `0 ... capacity - 1`: eviction slots
- `capacity`: no-op action

The byte-aware environment uses the same convention with `max_objects + 1`
actions, and can require multiple sequential eviction decisions for one miss.

## Fixed-slot experiments

Run the deterministic baseline benchmark:

```bash
python scripts/benchmark.py
```

Train the fixed-slot DQN:

```bash
python scripts/train_dqn.py \
    --episodes 80 \
    --requests 800 \
    --capacity 3 \
    --batch-size 64 \
    --seed 42
```

Compare the trained fixed-slot DQN with baselines:

```bash
python scripts/benchmark_rl.py \
    --checkpoint artifacts/dqn_best.pt \
    --capacity 3 \
    --requests 5000 \
    --seed 2026
```

Generate multi-seed stationary and drifting results:

```bash
python scripts/evaluate_generalization.py \
    --checkpoint artifacts/dqn_best.pt \
    --capacity 3 \
    --requests 5000 \
    --seed-start 20000 \
    --seed-count 10
```

Analyze paired reductions with bootstrap confidence intervals:

```bash
python scripts/analyze_generalization.py
```

Results are written to `artifacts/generalization_results.csv`.

## Byte-capacity experiments

The byte simulator enforces both a byte budget and a maximum object count.
Objects larger than the byte budget are served but bypass cache admission.
An incoming object can evict multiple resident objects.

Run the baseline comparison:

```bash
python scripts/benchmark_byte.py \
    --capacity-bytes 24576 \
    --requests 5000 \
    --seed 2026
```

Train the byte-aware DQN. The fixed feature scales are persisted in the
checkpoint as `byte_cache_v2` metadata:

```bash
python scripts/train_byte_dqn.py \
    --episodes 100 \
    --requests 1000 \
    --capacity-bytes 24576 \
    --max-objects 16 \
    --cost-scale-ms 250 \
    --size-scale-bytes 32768 \
    --batch-size 64 \
    --seed 42
```

Benchmark all byte-capacity policies:

```bash
python scripts/benchmark_byte_rl.py \
    --checkpoint artifacts/byte_dqn_best.pt \
    --capacity-bytes 24576 \
    --max-objects 16 \
    --requests 5000 \
    --seed 2026
```

Run multi-seed evaluation:

```bash
python scripts/evaluate_byte_generalization.py \
    --checkpoint artifacts/byte_dqn_best.pt \
    --capacity-bytes 24576 \
    --max-objects 16 \
    --requests 5000 \
    --seed-start 20000 \
    --seed-count 10
```

For an independent 30-seed report, use a seed range not used for training or
checkpoint selection:

```bash
python scripts/evaluate_byte_generalization.py \
    --checkpoint artifacts/byte_dqn_best.pt \
    --capacity-bytes 24576 \
    --max-objects 16 \
    --requests 5000 \
    --seed-start 30000 \
    --seed-count 30 \
    --output artifacts/byte_generalization_test_results.csv
```

The evaluator reports mean and standard deviation, paired bootstrap 95%
confidence intervals, per-seed win counts for miss cost and net cost, policy
overhead, hit ratio, evictions, peak bytes, and budget-violation flags.

The byte-DQN training script uses episode seeds beginning at the configured
training seed (default `42`) and validation seeds offset by `100000` and
`200000`. Reserve the independent test range beginning at `30000` for final
evaluation, and do not use those results to select a checkpoint.

Do not compare absolute miss costs from the fixed-slot and byte-capacity
benchmarks directly. They use different constraints and therefore represent
different experiments.

## Results recorded in this project

### Fixed-slot DQN

The recorded fixed-slot generalization analysis reported these aggregate mean
miss-cost reductions:

| Workload   | Compared with LRU | Compared with CostAware |
| ---------- | ----------------: | ----------------------: |
| Stationary |             36.9% |                   36.6% |
| Drifting   |             40.3% |                   15.9% |

These are descriptive aggregate comparisons, not proof of statistical
significance. Use `analyze_generalization.py` to inspect paired bootstrap
95% confidence intervals and per-seed wins.

### Byte-capacity baselines

The recorded 5,000-request drifting benchmark used a 24,576-byte budget and
reported:

| Policy    | Hit ratio | Miss cost (ms) | Evictions |
| --------- | --------: | -------------: | --------: |
| LRU       |     0.599 |      223,117.0 |     1,391 |
| LFU       |     0.659 |      273,401.0 |     1,091 |
| CostAware |     0.634 |      212,896.0 |     1,219 |

CostAware had the lowest miss cost in this baseline run. The byte-aware DQN
must be evaluated independently using `evaluate_byte_generalization.py`; its
success should be judged by paired miss-cost and net-cost comparisons, not by
hit ratio alone.

### Runtime/API demonstration

The FastAPI service was exercised successfully:

- `/health` returned `{"status":"ok"}`
- first `aggregation` request returned `X-Cache: MISS`
- second `aggregation` request returned `X-Cache: HIT`
- backend latency was approximately 180 ms on the miss and approximately 0 ms
  on the hit
- `search-index` returned `X-Cache-Bypassed: true` because it is larger than
  the 24 KiB cache budget

## FastAPI service

### SQLite-backed application

Create the database and profile first:

```bash
python scripts/init_demo_db.py
python scripts/profile_sqlite_origin.py \
    --database data/origin.sqlite \
    --repeats 15
```

Then start the application:

```bash
export ORIGIN_BACKEND=sqlite
export SQLITE_DATABASE="$PWD/data/origin.sqlite"
export ORIGIN_PROFILE="$PWD/configs/sqlite_profile.json"
export CACHE_POLICY=lru

uvicorn smartcache.api.app:app \
    --host 127.0.0.1 \
    --port 8000
```

Available endpoints:

```text
GET /
GET /health
GET /objects
GET /objects/{key}
GET /metrics
GET /runtime-info
GET /catalog
GET /metrics-ui
GET /simulation
GET /explorer
GET /docs
```

Browser pages:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/catalog
http://127.0.0.1:8000/metrics-ui
http://127.0.0.1:8000/explorer
```

The home page shows the active origin, policy, and cache capacity. The catalog
shows each cacheable object with profiled retrieval cost and response size;
clicking an object issues a real request through the cache. The metrics page
refreshes every second.

`/objects` and `/metrics` remain JSON APIs for scripts and monitoring. Live
metrics count only requests to `/objects/{key}`; opening UI pages does not
generate cache traffic. Metrics reset when the server restarts.

The Simulation Explorer replays one identical request trace through LRU, LFU,
CostAware, and the compatible fixed-slot DQN checkpoint. It shows each
request's hit/miss result, cache contents, evicted object, reward, policy
overhead, and cumulative cost. This replay is separate from the live API cache
and does not change `/metrics`.

The byte-aware DQN has a different observation/action space and should be
evaluated with the byte-capacity scripts rather than loaded into the fixed-slot
explorer.

Useful API checks:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/objects
curl http://127.0.0.1:8000/runtime-info
curl -i http://127.0.0.1:8000/objects/aggregation
curl -i http://127.0.0.1:8000/objects/aggregation
curl http://127.0.0.1:8000/metrics
```

The first repeated object request should be a miss and the second a hit. Then
open `/metrics-ui` or call `/metrics` to see the updated counters.

## Shared Redis cache

Install the optional runtime dependencies and start Redis:

```bash
python -m pip install -e ".[dev]"
docker run -d \
  --name smartcache-redis \
  --restart unless-stopped \
  -p 127.0.0.1:6379:6379 \
  redis:7-alpine
docker exec smartcache-redis redis-cli ping
```

Run the API with coordinated shared admission:

```bash
export CACHE_BACKEND=redis
export REDIS_URL=redis://127.0.0.1:6379/0
export CACHE_REDIS_NAMESPACE=smartcache:byte:v2
export CACHE_POLICY=dqn-byte
export CACHE_CHECKPOINT=artifacts/byte_dqn_best.pt

uvicorn smartcache.api.app:app --host 127.0.0.1 --port 8000
```

Redis is a startup requirement when selected; the API does not silently
pretend that a local cache is shared. The runtime performs a second lookup
after acquiring the admission lock to avoid duplicate admissions.

Run the reconciliation test:

```bash
REDIS_URL=redis://127.0.0.1:6379/15 pytest -q tests/test_redis_store.py
```

## HTTP load testing

With the API running, measure real HTTP behavior:

```bash
python scripts/load_test_http.py \
  --base-url http://127.0.0.1:8000 \
  --requests 5000 \
  --concurrency 20 \
  --output artifacts/http_load.json
```

Repeat with concurrency `1`, `10`, `50`, and `100`. The report records
completed responses, transport errors, error rate, throughput, p50, p95, p99,
HTTP status counts, and cache result headers.

## Checkpoint registry

The registry keeps versioned checkpoint copies, SHA-256 hashes, and an atomic
active manifest. Fixed-slot and byte-aware checkpoints remain separate:

```bash
python scripts/checkpoint_registry.py promote \
  --source artifacts/dqn_best.pt \
  --model-type fixed_slot
python scripts/checkpoint_registry.py promote \
  --source artifacts/byte_dqn_best.pt \
  --model-type byte
python scripts/checkpoint_registry.py status
python scripts/checkpoint_registry.py rollback --model-type byte
```

When `CACHE_CHECKPOINT` is unset, the API uses and verifies the active byte
registry entry when available. Set `CACHE_FALLBACK_POLICY=lru` or
`cost-aware` for explicit startup fallback. Runtime fallback events appear as
`policy_fallback_errors` in `/metrics`.

## Docker Compose

After creating `data/origin.sqlite` and `configs/sqlite_profile.json`:

```bash
docker compose up -d --build
curl -f http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/objects/aggregation
curl -i http://127.0.0.1:8000/objects/aggregation
docker compose down
```

## Docker

The Docker image includes the application source and a trained byte-aware
checkpoint:

```bash
docker build -t smart-cache-rl:latest .
docker run --rm \
    --name smart-cache-rl \
    -p 8001:8000 \
    smart-cache-rl:latest
```

The host uses port 8001 in this example so a local Uvicorn server can continue
using port 8000. Verify from another terminal:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/metrics
```

## How the comparison should be interpreted

The primary objective is minimizing total retrieval latency:

```text
net_cost_ms = total_miss_cost_ms + policy_time_ms
```

Lower policy overhead is better. A DQN is useful only when the miss-cost
savings exceed its inference overhead. Hit ratio is a supporting metric, not
the primary objective.

The strongest evidence comes from:

1. paired traces, where every policy sees exactly the same requests;
2. multiple independent seeds;
3. lower mean miss cost and net cost;
4. confidence intervals for paired reductions;
5. inference overhead that is small relative to latency saved;
6. byte-budget invariants that are never violated.
