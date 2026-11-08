from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from smartcache.core.interfaces import CacheEntry


def encode_byte_state(
    entries: Sequence[CacheEntry],
    incoming_key: str,
    catalog: Mapping[str, Mapping[str, float]],
    frequency_counts: Mapping[str, int],
    position: int,
    capacity_bytes: int,
    used_bytes: int,
    max_objects: int,
    frequency_window: int,
    recency_window: int,
    cost_scale_ms: float,
    size_scale_bytes: float,
    pending_cost_charged: bool,
) -> np.ndarray:
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    if len(entries) > max_objects:
        raise ValueError("Cache contains more entries than max_objects")
    if capacity_bytes < 1:
        raise ValueError("capacity_bytes must be positive")
    if frequency_window < 1 or recency_window < 1:
        raise ValueError("Window sizes must be positive")
    if incoming_key not in catalog:
        raise KeyError(f"Unknown incoming object: {incoming_key}")
    if cost_scale_ms <= 0 or size_scale_bytes <= 0:
        raise ValueError("Feature scales must be positive")

    def normalize(value: float, maximum: float) -> float:
        denominator = np.log1p(max(1.0, float(maximum)))
        return float(np.log1p(max(0.0, float(value))) / denominator)

    def frequency(key: str) -> float:
        count = max(0, int(frequency_counts.get(key, 0)))
        return min(1.0, float(np.log1p(count) / np.log1p(frequency_window)))

    features: list[float] = []
    for entry in entries:
        age = max(0, position - entry.last_access)
        recency = max(0.0, 1.0 - age / recency_window)
        metadata = catalog[entry.key]
        features.extend(
            [
                recency,
                frequency(entry.key),
                normalize(metadata["cost"], cost_scale_ms),
                normalize(metadata["size"], size_scale_bytes),
                1.0,
            ]
        )
    features.extend([0.0] * ((max_objects - len(entries)) * 5))
    incoming = catalog[incoming_key]
    features.extend(
        [
            normalize(incoming["cost"], cost_scale_ms),
            normalize(incoming["size"], size_scale_bytes),
            frequency(incoming_key),
            min(1.0, used_bytes / capacity_bytes),
            float(pending_cost_charged),
        ]
    )
    return np.asarray(features, dtype=np.float32)
