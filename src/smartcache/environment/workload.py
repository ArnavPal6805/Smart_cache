from __future__ import annotations

import random


OBJECTS = [
    {"key": "metadata", "cost": 2.0, "size": 512},
    {"key": "user-profile", "cost": 4.0, "size": 2048},
    {"key": "api-response", "cost": 15.0, "size": 4096},
    {"key": "aggregation", "cost": 180.0, "size": 8192},
    {"key": "embedding", "cost": 250.0, "size": 16384},
    {"key": "search-index", "cost": 90.0, "size": 32768},
    {"key": "report", "cost": 120.0, "size": 4096},
]

PHASE_A_WEIGHTS = [0.45, 0.18, 0.13, 0.10, 0.08, 0.04, 0.02]
PHASE_B_WEIGHTS = [0.05, 0.08, 0.07, 0.25, 0.25, 0.20, 0.10]


def generate_workload(
    number_of_requests: int = 2000,
    seed: int = 42,
    drift: bool = False,
) -> list[dict]:
    if number_of_requests < 1:
        raise ValueError("number_of_requests must be positive")
    rng = random.Random(seed)
    if not drift:
        selected = rng.choices(OBJECTS, weights=PHASE_A_WEIGHTS, k=number_of_requests)
    else:
        transition = number_of_requests // 2
        selected = [
            *rng.choices(OBJECTS, weights=PHASE_A_WEIGHTS, k=transition),
            *rng.choices(
                OBJECTS,
                weights=PHASE_B_WEIGHTS,
                k=number_of_requests - transition,
            ),
        ]
    return [dict(request) for request in selected]
