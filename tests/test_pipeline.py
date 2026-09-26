import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from conftest import colour_image
from maxvit_classification_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    IMAGENET_GROUPS,
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MaxViTPipeline,
    classification_metrics,
    evaluation_report,
    is_pinned,
    majority_class,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)
from maxvit_classification_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "weights" / MODEL_KEY


def test_identity_constants_agree_with_the_committed_manifest():
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert MODEL_ID == "timm/maxvit_tiny_tf_224.in1k" == manifest["modelId"]
    assert [r["path"] for r in manifest["referenceFiles"]] == ["pytorch_model.bin"]
    assert manifest["revision"] == pipeline_module.MODEL_REVISION
    assert is_pinned() == bool(HEX40.match(pipeline_module.MODEL_REVISION))
    assert DEFAULT_WEIGHTS_DIR == SNAPSHOT and ARTIFACT_FORMAT == "maxvit-adapter-v1"
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert {entry["path"] for entry in manifest["files"]} == {"README.md", "config.json", "model.safetensors"}
    if not is_pinned():
        assert all(entry["sha256"] is None for entry in manifest["files"])


def test_committed_config_matches_the_manifest_size_and_timm_pretrained_cfg():
    raw = (SNAPSHOT / "config.json").read_bytes()
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert len(raw) == next(e["bytes"] for e in manifest["files"] if e["path"] == "config.json")
    config = json.loads(raw)
    assert (config["architecture"], config["pretrained_cfg"]["tag"]) == (
        pipeline_module.ARCHITECTURE,
        pipeline_module.PRETRAINED_TAG,
    )
    assert (
        config["num_classes"] == 1000
        and config["pretrained_cfg"]["classifier"] == f"{pipeline_module.HEAD_MODULE}.fc"
    )
    timm = pytest.importorskip("timm")
    builtin = timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=False).pretrained_cfg
    assert (
        config["pretrained_cfg"]["fixed_input_size"] is True and config["pretrained_cfg"]["crop_pct"] == 0.95
    )
    for key in ("input_size", "interpolation", "crop_pct", "crop_mode", "mean", "std"):
        assert (
            tuple(builtin[key]) == tuple(config["pretrained_cfg"][key])
            if isinstance(builtin[key], tuple | list)
            else builtin[key] == config["pretrained_cfg"][key]
        )


def test_imagenet_groups_name_frog_and_big_truck_classes():
    timm = pytest.importorskip("timm")
    labels = pipeline_module.imagenet_labels()
    assert len(labels) == 1000 and timm is not None
    assert all("frog" in labels[i] for i in IMAGENET_GROUPS["frog"])
    assert [labels[i].split(",")[0] for i in IMAGENET_GROUPS["truck"]] == [
        "fire engine", "garbage truck", "moving van", "tow truck", "trailer truck",
    ]  # fmt: skip
    assert 717 not in IMAGENET_GROUPS["truck"] and labels[717].startswith(
        "pickup"
    )  # CIFAR-10 excludes pickups


def test_unpinned_package_refuses_every_weight_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    for call in (
        lambda: verify_snapshot(tmp_path),
        lambda: stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None),
        lambda: MaxViTPipeline.from_pretrained(weights_dir=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="pin_snapshot.py"):
            call()


def _write_snapshot(
    root: Path, revision: str, content: bytes, sha: str | None = None, size: int | None = None
) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_and_rejects_every_mismatch(tmp_path, pinned):
    _write_snapshot(tmp_path, pinned, b"{}")
    assert verify_snapshot(tmp_path)["files"] == 1
    _write_snapshot(tmp_path, pinned, b"{}", sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, "f" * 40, b"{}")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"][0]["sha256"] = None
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, pinned):
    payload = b"weights"
    _write_snapshot(tmp_path, pinned, b"{}")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"].append(
        {"path": "model.safetensors", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake(relative, root):
        fetched.append(relative)
        (root / relative).write_bytes(payload)

    assert (
        stage_missing_files(tmp_path, allow_download=True, downloader=fake)
        == ["model.safetensors"]
        == fetched
    )
    assert verify_snapshot(tmp_path)["weights_sha256"] == hashlib.sha256(payload).hexdigest()
    manifest["modelId"] = "someone/else"
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=fake)


def test_classification_metrics_and_majority_baseline():
    metrics = classification_metrics(
        ["frog", "frog", "truck", "frog"],
        ["frog", "truck", "truck", "truck"],
        ["frog", "truck"],
        majority_class="truck",
    )
    assert metrics["accuracy"] == 0.5 and metrics["per_class_recall"] == {"frog": 1.0, "truck": 1 / 3}
    assert metrics["balanced_accuracy"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert metrics["confusion_matrix"]["rows_true_cols_predicted"] == [[1, 0], [2, 1]]
    assert metrics["majority_baseline_accuracy"] == 0.75
    assert classification_metrics(["a"], ["a"], ["a", "b"])["per_class_recall"]["b"] is None
    with pytest.raises(ValueError, match="majority_class"):
        classification_metrics(["a"], ["a"], ["a", "b"], majority_class="c")
    with pytest.raises(ValueError, match="predictions"):
        classification_metrics(["a"], [], ["a", "b"])
    assert majority_class([{"label": "b"}, {"label": "a"}, {"label": "b"}]) == "b"
    assert majority_class([{"label": "b"}, {"label": "a"}]) == "a"


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs(
        [colour_image("frog", 0), Image.new("L", (40, 20))], top_k=3, names=["a.png", "b.png"]
    )
    assert manifest["schema"] == INPUT_SCHEMA and manifest["verdict"] == "accepted" and manifest["top_k"] == 3
    assert manifest["inputs"][1] == {"id": "b.png", "mode": "L", "size": [40, 20]}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    for images, top_k, error, message in (
        ("x.png", 5, TypeError, "PIL.Image.Image"),
        ([Image.new("RGB", (MIN_IMAGE_SIDE - 1, 20))], 5, ValueError, "outside"),
        ([Image.new("RGB", (MAX_IMAGE_SIDE + 1, 20))], 5, ValueError, "outside"),
        ([colour_image("frog", 0)] * (MAX_BATCH + 1), 5, ValueError, "MAX_BATCH"),
        ([colour_image("frog", 0)], 0, ValueError, "top_k"),
        ([colour_image("frog", 0)], True, TypeError, "top_k"),
    ):
        with pytest.raises(error, match=message):
            validate_inputs(images, top_k)
    with pytest.raises(ValueError, match="one entry per image"):
        validate_inputs([colour_image("frog", 0)], names=["a", "b"])


def _result(indices):
    return {
        "predictions": [
            {"top_k": [{"index": i, "label": str(i), "score": 0.5} for i in row]} for row in indices
        ],
        "top_k": 2,
    }


def test_evaluation_report_verdicts():
    report = evaluation_report(_result([[30, 1]]))
    assert (
        report["verdict"] == "not-measurable"
        and report["metrics"] == []
        and "majority-class" in report["needs"]
    )
    report = evaluation_report(
        _result([[30, 1], [1, 867], [5, 6]]), ["frog", "truck", "truck"], groups=IMAGENET_GROUPS
    )
    assert report["verdict"] == "sample-sanity"
    assert [(m["k"], m["value"]) for m in report["metrics"]] == [
        (1, pytest.approx(1 / 3)),
        (2, pytest.approx(2 / 3)),
    ]
    with pytest.raises(ValueError, match="labels"):
        evaluation_report(_result([[30, 1]]), ["frog", "truck"], groups=IMAGENET_GROUPS)
