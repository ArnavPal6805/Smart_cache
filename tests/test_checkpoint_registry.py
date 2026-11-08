import json
import importlib.util
from pathlib import Path


def load_registry_module():
    path = Path(__file__).parents[1] / "scripts" / "checkpoint_registry.py"
    spec = importlib.util.spec_from_file_location("checkpoint_registry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint_registry = load_registry_module()


def test_registry_promotes_and_rolls_back_checkpoints(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    monkeypatch.setattr(checkpoint_registry, "REGISTRY", registry)
    monkeypatch.setattr(checkpoint_registry, "MANIFEST", registry / "active.json")

    source = checkpoint_registry.ROOT / "artifacts" / "byte_dqn_best.pt"
    checkpoint_registry.promote(source, "byte")
    first = json.loads((registry / "active.json").read_text())

    checkpoint_registry.promote(source, "byte")
    checkpoint_registry.rollback("byte")
    restored = json.loads((registry / "active.json").read_text())

    assert restored["active"]["byte"]["version"] == first["active"]["byte"]["version"]
    assert restored["active"]["byte"]["model_type"] == "byte"
