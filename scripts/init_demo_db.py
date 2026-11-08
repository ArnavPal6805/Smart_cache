from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path

CATEGORIES = [
    "electronics", "books", "clothing", "home", "sports", "beauty",
    "toys", "automotive", "food", "office", "garden", "health",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the SmartCache SQLite origin.")
    parser.add_argument("--database", default="data/origin.sqlite")
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1_000:
        parser.error("--rows must be at least 1000")

    path = Path(args.database)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS purchases;
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                rating REAL NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE purchases (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                price REAL NOT NULL
            );
            CREATE INDEX products_category_idx ON products(category);
            CREATE INDEX purchases_user_idx ON purchases(user_id);
            """
        )
        products = []
        for product_id in range(1, args.rows + 1):
            category = rng.choice(CATEGORIES)
            price = round(rng.uniform(5.0, 1500.0), 2)
            rating = round(rng.uniform(1.0, 5.0), 2)
            products.append(
                (
                    product_id,
                    f"{category.title()} product {product_id}",
                    category,
                    price,
                    rating,
                    f"Deterministic description for {category} product {product_id}.",
                )
            )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products
        )
        purchases = []
        for purchase_id in range(1, max(1_000, args.rows // 2) + 1):
            product_id = rng.randint(1, args.rows)
            user_id = rng.randint(1, max(100, args.rows // 20))
            price = products[product_id - 1][3]
            purchases.append((purchase_id, user_id, product_id, price))
        connection.executemany("INSERT INTO purchases VALUES (?, ?, ?, ?)", purchases)
        connection.commit()
    print(f"created {path} with {args.rows} products")


if __name__ == "__main__":
    main()
