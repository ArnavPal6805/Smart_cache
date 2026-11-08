from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def encode_state(
    slot_keys: Sequence[str | None],
    incoming_key: str,
    catalog: Mapping[str, Mapping[str, float]],
    last_access: Mapping[str, int],
    frequency_counts: Mapping[str, int],
    position: int,
    frequency_window: int,
    recency_window: int,
    max_cost: float,
    max_size: float,
) -> np.ndarray:
    """Encode cache slots and the incoming request as a DQN state vector."""
    if frequency_window < 1 or recency_window < 1:
        raise ValueError("Window sizes must be positive")
    if incoming_key not in catalog:
        raise KeyError(f"Unknown incoming key: {incoming_key}")
    if max_cost <= 0 or max_size <= 0:
        raise ValueError("Normalization maxima must be positive")

    def normalize(value: float, maximum: float) -> float:
        denominator = np.log1p(max(1.0, maximum))
        return float(np.log1p(max(0.0, value)) / denominator)

    def recency(key: str) -> float:
        age = max(0, position - last_access.get(key, position))
        return float(np.exp(-age / recency_window))

    def frequency(key: str) -> float:
        return min(1.0, max(0, frequency_counts.get(key, 0)) / frequency_window)

    values: list[float] = []
    for key in slot_keys:
        if key is None:
            values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        item = catalog[key]
        values.extend(
            (
                recency(key),
                frequency(key),
                normalize(float(item["cost"]), max_cost),
                normalize(float(item["size"]), max_size),
                1.0,
            )
        )

    incoming = catalog[incoming_key]
    values.extend(
        (
            normalize(float(incoming["cost"]), max_cost),
            normalize(float(incoming["size"]), max_size),
            frequency(incoming_key),
        )
    )
    return np.asarray(values, dtype=np.float32)
