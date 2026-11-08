from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from smartcache.api.demo_origin import OriginSpec

OBJECT_KEYS = (
    "metadata",
    "user-profile",
    "api-response",
    "aggregation",
    "embedding",
    "report",
    "search-index",
)


@dataclass(frozen=True)
class SQLiteOriginResult:
    payload: bytes
    size_bytes: int


class SQLiteOrigin:
    """Read-only origin adapter exposing deterministic cacheable SQL queries."""

    def __init__(self, database_path: str | Path):
        path = Path(database_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.database_path = str(path.resolve())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def load(self, key: str) -> bytes:
        if key not in OBJECT_KEYS:
            raise KeyError(f"Unknown origin object: {key}")
        with self._connect() as connection:
            payload = getattr(self, f"_load_{key.replace('-', '_')}")(connection)
        return payload

    def _load_metadata(self, connection: sqlite3.Connection) -> bytes:
        row = connection.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MIN(price) AS min_price,
                   MAX(price) AS max_price,
                   AVG(price) AS average_price
            FROM products
            """
        ).fetchone()
        return json.dumps(dict(row), sort_keys=True).encode()

    def _load_user_profile(self, connection: sqlite3.Connection) -> bytes:
        rows = connection.execute(
            """
            SELECT user_id, COUNT(*) AS purchases, SUM(price) AS total_spend
            FROM purchases
            GROUP BY user_id
            ORDER BY total_spend DESC, user_id
            LIMIT 10
            """
        ).fetchall()
        return json.dumps([dict(row) for row in rows], sort_keys=True).encode()

    def _load_api_response(self, connection: sqlite3.Connection) -> bytes:
        rows = connection.execute(
            """
            SELECT id, category, price, rating
            FROM products
            ORDER BY id
            LIMIT 100
            """
        ).fetchall()
        return json.dumps([dict(row) for row in rows], separators=(",", ":")).encode()

    def _load_aggregation(self, connection: sqlite3.Connection) -> bytes:
        rows = connection.execute(
            """
            SELECT category, COUNT(*) AS products, ROUND(AVG(price), 4) AS average_price,
                   ROUND(AVG(rating), 4) AS average_rating
            FROM products
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
        return json.dumps([dict(row) for row in rows], sort_keys=True).encode()

    def _load_embedding(self, connection: sqlite3.Connection) -> bytes:
        row = connection.execute(
            "SELECT category, AVG(price) AS average_price, AVG(rating) AS average_rating "
            "FROM products GROUP BY category ORDER BY category"
        ).fetchall()
        values = []
        for item in row:
            digest = hashlib.sha256(str(item["category"]).encode()).digest()
            values.extend(
                [
                    math.tanh(float(item["average_price"]) / 100.0),
                    math.tanh(float(item["average_rating"])),
                    digest[0] / 255.0,
                ]
            )
        return b"".join(struct.pack("<f", value) for value in values)

    def _load_report(self, connection: sqlite3.Connection) -> bytes:
        rows = connection.execute(
            """
            SELECT category, COUNT(*) AS products, SUM(price) AS revenue
            FROM products
            GROUP BY category
            ORDER BY revenue DESC
            """
        ).fetchall()
        report = {
            "title": "Product revenue report",
            "categories": [dict(row) for row in rows],
        }
        return json.dumps(report, sort_keys=True).encode()

    def _load_search_index(self, connection: sqlite3.Connection) -> bytes:
        rows = connection.execute(
            """
            SELECT id, name, category, description
            FROM products
            ORDER BY id
            """
        ).fetchall()
        return json.dumps([dict(row) for row in rows], separators=(",", ":")).encode()

    def loader(self) -> Callable[[str], bytes]:
        return self.load

    def specs_from_profile(self, profile_path: str | Path | None = None) -> dict[str, OriginSpec]:
        profile: dict = {}
        if profile_path is not None and Path(profile_path).is_file():
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        configured = profile.get("objects", {})
        specs: dict[str, OriginSpec] = {}
        for key in OBJECT_KEYS:
            payload = self.load(key)
            item = configured.get(key, {})
            specs[key] = OriginSpec(
                retrieval_cost_ms=max(0.001, float(item.get("retrieval_cost_ms", 1.0))),
                size_bytes=int(item.get("size_bytes", len(payload))),
                fill_byte=payload[0] if payload else 0,
            )
        return specs
