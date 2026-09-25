"""End-to-end adaptation on a shrunken random-weight MaxViT: no checkpoint, no network.

The model is timm's ``maxvit_tiny_tf_224`` architecture, the one the pinned checkpoint loads into, with
one block per stage at width 32 (MaxViT's input size stays fixed at 224) so a CPU step takes under a
second. These tests exercise the real prediction, zero-shot, fine-tuning, evaluation, adapter export and
adapter reload code paths. They say nothing about classification quality.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from safetensors.torch import save_file  # noqa: E402

from conftest import colour_records  # noqa: E402
from maxvit_classification_pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    IMAGENET_GROUPS,
    MODEL_ID,
    MODEL_KEY,
    SAMPLE_CLASSES,
    MaxViTPipeline,
    build_model,
    imagenet_labels,
    majority_class,
    split_dataset,
)
from maxvit_classification_pipeline import pipeline as pipeline_module  # noqa: E402

SMALL = {"embed_dim": (32, 32, 32, 32), "depths": (1, 1, 1, 1), "stem_width": 16, "head_hidden_size": 32}


def _imagenet_pipeline(seed: int = 0) -> MaxViTPipeline:
    torch.manual_seed(seed)
    model = build_model(**SMALL)
    return MaxViTPipeline.from_components(
        model, class_names=imagenet_labels(), transform=pipeline_module.eval_transform(model)
    )


def _adapter_pipeline(seed: int = 0, class_names=SAMPLE_CLASSES) -> MaxViTPipeline:
    pipe = _imagenet_pipeline(seed)
    torch.manual_seed(seed + 1)
    pipe.reinitialised = pipeline_module._rehead(pipe.model, len(class_names))
    pipe.class_names = tuple(class_names)
    return pipe


@pytest.fixture(scope="module")
def data():
    return split_dataset(colour_records(8), train_fraction=0.75, seed=0)


def test_build_model_matches_the_published_parameter_count():
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 30_916_528  # the Hub card's "Params (M): 30.9"
    assert pipeline_module._backbone_prefixes(model) == ("stem.", "stages.", "norm.")
    assert pipeline_module._rehead(model, 2) == ("head.fc.",) and model.head.fc.out_features == 2


def test_predict_and_zero_shot_on_the_imagenet_head(data):
    _train, held = data
    pipe = _imagenet_pipeline()
    out = pipe.predict([r["image"] for r in held], top_k=5)
    assert len(out["predictions"]) == len(held) and len(out["predictions"][0]["top_k"]) == 5
    scores = [item["score"] for item in out["predictions"][0]["top_k"]]
    assert scores == sorted(scores, reverse=True) and 0 < sum(scores) <= 1
    zero = pipe.zero_shot_evaluate(held, IMAGENET_GROUPS, majority="frog")
    assert (
        0 <= zero["accuracy"] <= 1
        and zero["majority_baseline_accuracy"] == 0.5
        and 0 < zero["mean_group_mass"] < 1
    )
    with pytest.raises(ValueError, match="no ImageNet group"):
        pipe.zero_shot_evaluate([{"image": held[0]["image"], "label": "cat"}])
    with pytest.raises(RuntimeError, match="1000-class"):
        _adapter_pipeline().zero_shot_evaluate(held)


def test_full_finetune_evaluate_export_and_reload(tmp_path, data):
    train, held = data
    pipe = _adapter_pipeline()
    baseline = pipe.evaluate(held, majority=majority_class(train))
    assert baseline["adapted"] is False and baseline["majority_baseline_accuracy"] == 0.5
    run = pipe.finetune(train, epochs=2, batch_size=4, learning_rate=1e-3, seed=1)
    assert len(run["epoch_losses"]) == 2 and run["trainable_parameters"] == run["total_parameters"]
    assert run["frozen_prefixes"] == [] and torch.isfinite(torch.tensor(run["final_loss"]))
    adapted = pipe.evaluate(held, majority=majority_class(train))
    assert adapted["adapted"] and set(adapted) >= {
        "accuracy",
        "balanced_accuracy",
        "per_class_recall",
        "confusion_matrix",
    }

    path = tmp_path / "adapter.safetensors"
    descriptor = pipe.save_artifact(path, notes="small")
    assert descriptor["format"] == ARTIFACT_FORMAT and descriptor["tensors"] == len(pipe.model.state_dict())
    fresh = _adapter_pipeline(
        seed=7
    )  # different weights everywhere: a full adapter must overwrite all of them
    fresh.apply_artifact(path)
    assert all(torch.equal(v, fresh.model.state_dict()[k]) for k, v in pipe.model.state_dict().items())
    images = [r["image"] for r in held]
    assert pipe.predict(images)["predictions"] == fresh.predict(images)["predictions"]


def test_frozen_finetune_keeps_backbone_weights_and_batchnorm_statistics(tmp_path, data):
    train, held = data
    pipe = _adapter_pipeline()
    head = f"{pipeline_module.HEAD_MODULE}."
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    run = pipe.finetune(train, epochs=1, batch_size=4, learning_rate=1e-3, seed=1, freeze_backbone=True)
    after = pipe.model.state_dict()
    frozen = [k for k in before if not k.startswith(head)]
    assert any("running_mean" in k for k in frozen)
    assert all(torch.equal(before[k], after[k]) for k in frozen)
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(head))
    assert 0 < run["trainable_parameters"] < run["total_parameters"] and head not in run["frozen_prefixes"]

    path = tmp_path / "head.safetensors"
    descriptor = pipe.save_artifact(path)
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        assert all(name.startswith(head) for name in handle.keys())  # noqa: SIM118 -- safe_open is not a dict
    fresh = _adapter_pipeline()  # same seed: identical frozen backbone
    fresh.apply_artifact(path)
    images = [r["image"] for r in held]
    assert (
        pipe.predict(images)["predictions"] == fresh.predict(images)["predictions"]
        and descriptor["tensors"] == len([k for k in pipe.model.state_dict() if k.startswith(head)]) == 6
    )


def test_frozen_finetune_without_the_batchnorm_hold_would_drift(monkeypatch, data):
    train, _held = data
    monkeypatch.setattr(pipeline_module, "_hold_batchnorm", lambda modules: None)
    pipe = _adapter_pipeline()
    before = {k: v.clone() for k, v in pipe.model.state_dict().items() if "running_mean" in k}
    pipe.finetune(train, epochs=1, batch_size=4, seed=1, freeze_backbone=True)
    assert any(not torch.equal(v, pipe.model.state_dict()[k]) for k, v in before.items())


def test_apply_artifact_refuses_mismatched_adapters(tmp_path, data):
    train, _held = data
    pipe = _adapter_pipeline()
    pipe.finetune(train[:2] + train[-2:], epochs=1, batch_size=4, seed=1, freeze_backbone=True)
    path = tmp_path / "adapter.safetensors"
    pipe.save_artifact(path)
    with pytest.raises(ValueError, match="class_names differ"):
        _adapter_pipeline(class_names=("a", "b")).apply_artifact(path)
    tensor = {"head.fc.bias": torch.zeros(2)}
    forged = tmp_path / "forged.safetensors"
    save_file(tensor, str(forged), metadata={"format": "other", "model_id": MODEL_ID, "model_key": MODEL_KEY})
    with pytest.raises(ValueError, match="artifact format"):
        _adapter_pipeline().apply_artifact(forged)
    partial = tmp_path / "partial.safetensors"
    metadata = {
        "format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": pipeline_module.MODEL_REVISION,
        "model_key": MODEL_KEY,
        "class_names": json.dumps(list(SAMPLE_CLASSES)),
        "frozen_prefixes": json.dumps([]),
        "base_state_digest": "",
        "notes": "",
    }
    save_file(tensor, str(partial), metadata=metadata)
    with pytest.raises(ValueError, match="missing trainable tensors"):
        _adapter_pipeline().apply_artifact(partial)
    with pytest.raises(RuntimeError, match="not been fine-tuned"):
        _adapter_pipeline().save_artifact(tmp_path / "none.safetensors")


def test_finetune_rejects_bad_hyperparameters_and_records(data):
    train, _held = data
    pipe = _adapter_pipeline()
    with pytest.raises(ValueError, match="batch_size"):
        pipe.finetune(train, batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        pipe.finetune(train, learning_rate=0)
    with pytest.raises(ValueError, match="unknown class"):
        pipe.finetune([{**train[0], "label": "cat"}, *train])
    with pytest.raises(ValueError, match="outside this pipeline's classes"):
        pipe.evaluate([{**train[0], "label": "cat"}])
