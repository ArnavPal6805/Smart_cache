from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from smartcache.training.dqn_agent import DQNAgent

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts" / "model_registry"
MANIFEST = REGISTRY / "active.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST.exists():
        return {"active": {}, "history": []}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def validate_checkpoint(path: Path, model_type: str) -> dict:
    agent, metadata = DQNAgent.load_checkpoint(path, device="cpu")
    if agent.state_dim <= 0 or agent.action_dim <= 0:
        raise ValueError("Checkpoint dimensions must be positive")
    if model_type == "byte":
        if metadata.get("feature_version") != "byte_cache_v2":
            raise ValueError("Byte registry accepts only byte_cache_v2 checkpoints")
        max_objects = int(metadata["max_objects"])
        expected_state_dim = max_objects * 5 + 5
        if agent.state_dim != expected_state_dim or agent.action_dim != max_objects + 1:
            raise ValueError("Byte checkpoint dimensions do not match metadata")
    return {
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "feature_version": metadata.get("feature_version"),
    }


def write_manifest(value: dict) -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, MANIFEST)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def promote(source: Path, model_type: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint_info = validate_checkpoint(source, model_type)
    digest = sha256(source)
    version = f"{model_type}-{time.strftime('%Y%m%d-%H%M%S')}-{digest[:12]}"
    destination = REGISTRY / version / source.name
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, destination)
    manifest = read_manifest()
    previous = manifest.get("active", {}).get(model_type)
    if previous:
        previous = {**previous, "model_type": model_type}
        manifest.setdefault("history", []).append(previous)
    manifest.setdefault("active", {})[model_type] = {
        "version": version,
        "filename": source.name,
        "sha256": digest,
        "path": str(destination.relative_to(ROOT)),
        "model_type": model_type,
        "checkpoint": checkpoint_info,
    }
    write_manifest(manifest)
    print(json.dumps(manifest["active"][model_type], indent=2))


def rollback(model_type: str) -> None:
    manifest = read_manifest()
    candidates = [
        item for item in manifest.get("history", [])
        if item and item.get("model_type", model_type) == model_type
    ]
    if not candidates:
        raise RuntimeError(f"No rollback version available for {model_type}")
    previous = candidates.pop()
    manifest["active"][model_type] = previous
    manifest["history"] = candidates
    write_manifest(manifest)
    print(json.dumps(previous, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--source", type=Path, required=True)
    promote_parser.add_argument("--model-type", choices=["fixed_slot", "byte"], required=True)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--model-type", choices=["fixed_slot", "byte"], required=True)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "promote":
        promote(args.source.resolve(), args.model_type)
    elif args.command == "rollback":
        rollback(args.model_type)
    else:
        print(json.dumps(read_manifest(), indent=2))


if __name__ == "__main__":
    main()
