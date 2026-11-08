import json
import sqlite3

from smartcache.api.sqlite_origin import OBJECT_KEYS, SQLiteOrigin


def create_database(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                price REAL,
                rating REAL,
                description TEXT
            );
            CREATE TABLE purchases (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product_id INTEGER,
                price REAL
            );
            """
        )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "Book", "books", 10.0, 4.5, "A book"),
                (2, "Phone", "electronics", 100.0, 4.0, "A phone"),
            ],
        )
        connection.executemany(
            "INSERT INTO purchases VALUES (?, ?, ?, ?)",
            [(1, 1, 1, 10.0), (2, 1, 2, 100.0)],
        )


def test_sqlite_origin_loads_all_objects(tmp_path):
    database = tmp_path / "origin.sqlite"
    create_database(database)
    origin = SQLiteOrigin(database)

    payloads = {key: origin.load(key) for key in OBJECT_KEYS}

    assert all(payloads.values())
    assert json.loads(payloads["metadata"])["row_count"] == 2
    assert len(payloads["embedding"]) % 4 == 0


def test_sqlite_origin_profile_uses_measured_payload_sizes(tmp_path):
    database = tmp_path / "origin.sqlite"
    create_database(database)
    origin = SQLiteOrigin(database)

    specs = origin.specs_from_profile()

    assert set(specs) == set(OBJECT_KEYS)
    assert specs["api-response"].size_bytes == len(origin.load("api-response"))
