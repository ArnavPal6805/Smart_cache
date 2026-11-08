from __future__ import annotations

from dataclasses import dataclass
from time import sleep


@dataclass(frozen=True)
class OriginSpec:
    retrieval_cost_ms: float
    size_bytes: int
    fill_byte: int


ORIGIN_SPECS = {
    "metadata": OriginSpec(2.0, 512, ord("M")),
    "user-profile": OriginSpec(4.0, 2048, ord("U")),
    "api-response": OriginSpec(15.0, 4096, ord("A")),
    "aggregation": OriginSpec(180.0, 8192, ord("G")),
    "embedding": OriginSpec(250.0, 16384, ord("E")),
    "search-index": OriginSpec(90.0, 32768, ord("S")),
    "report": OriginSpec(120.0, 4096, ord("R")),
}


def load_origin_payload(key: str) -> bytes:
    spec = ORIGIN_SPECS[key]
    sleep(spec.retrieval_cost_ms / 1000.0)
    return bytes([spec.fill_byte]) * spec.size_bytes
