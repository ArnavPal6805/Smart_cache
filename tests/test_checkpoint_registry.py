import json

from scripts import checkpoint_registry


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
