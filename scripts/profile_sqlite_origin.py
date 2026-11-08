from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from smartcache.api.sqlite_origin import OBJECT_KEYS, SQLiteOrigin


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SQLite origin objects.")
    parser.add_argument("--database", default="data/origin.sqlite")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", default="configs/sqlite_profile.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    origin = SQLiteOrigin(args.database)
    objects = {}
    for key in OBJECT_KEYS:
        timings_ms = []
        payload = b""
        for _ in range(args.repeats):
            started = perf_counter_ns()
            payload = origin.load(key)
            timings_ms.append((perf_counter_ns() - started) / 1_000_000)
        objects[key] = {
            "retrieval_cost_ms": round(float(median(timings_ms)), 6),
            "size_bytes": len(payload),
            "timings_ms": [round(value, 6) for value in timings_ms],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"database": str(Path(args.database).resolve()), "objects": objects},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"profiled {len(objects)} objects -> {output}")


if __name__ == "__main__":
    main()
