import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from types import SimpleNamespace

from ml.multimodal import (MultimodalCVDModel, cross_modal_infonce, gradcam_1d,
                           gradcam_2d, load_multimodal_checkpoint,
                           mc_dropout_predict, save_multimodal_checkpoint)
from ml.multimodal_data import align_patient_modalities, load_ecg, load_medical_image
from ml.train_multimodal import train as train_multimodal
from ml.multimodal_evaluation import evaluate_logits, load_paired_evaluation
from ml.subgroups_multimodal import evaluate_multimodal_subgroups


def test_patient_alignment_and_readers(tmp_path):
    np.savetxt(tmp_path / "p1.csv", np.sin(np.arange(50)), delimiter=",")
    Image.fromarray(np.full((24, 32, 3), 100, dtype=np.uint8)).save(tmp_path / "p1.png")
    clinical = pd.DataFrame({"patient_id": [1, 2], "label": [0, 1], "age": [40, 65]})
    ecg = pd.DataFrame({"patient_id": ["1", "x"], "ecg_path": ["p1.csv", "p1.csv"]})
    images = pd.DataFrame({"patient_id": [1], "image_path": ["p1.png"]})
    aligned, report = align_patient_modalities(clinical, ecg, images,
                                               ecg_manifest_dir=tmp_path, image_manifest_dir=tmp_path)
    assert report["complete_triplets"] == 1
    assert report["missing_image_patient_ids"] == ["2"]
    assert report["unmatched_ecg_patient_ids"] == ["x"]
    assert load_ecg(aligned.loc[0, "ecg_path"], samples=128).shape == (128,)
    assert load_medical_image(aligned.loc[0, "image_path"], size=32).shape == (3, 32, 32)
    duplicated = pd.concat([ecg.iloc[:1], ecg.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Multiple ecg files"):
        align_patient_modalities(clinical, duplicated, images,
                                 ecg_manifest_dir=tmp_path, image_manifest_dir=tmp_path)


def test_neural_model_infonce_uncertainty_cam_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = MultimodalCVDModel(clinical_dim=5, embedding_dim=16)
    model.eval()
    clinical = torch.randn(3, 5)
    ecg = torch.randn(3, 512)
    image = torch.randn(3, 3, 64, 64)
    logits, embeddings = model(clinical, ecg, image)
    assert logits.shape == (3, 2)
    loss = cross_modal_infonce(embeddings, ["a", "b", "c"])
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="patient"):
        cross_modal_infonce(embeddings, ["a", "a", "c"])
    uncertainty = mc_dropout_predict(model, {"clinical": clinical, "ecg": ecg, "image": image}, passes=6)
    assert uncertainty["mean_probability"].shape == (3, 2)
    assert torch.all(uncertainty["variance"] >= 0)
    assert gradcam_1d(model, ecg, clinical=clinical, image=image).shape == (3, 512)
    assert gradcam_2d(model, image, clinical=clinical, ecg=ecg).shape == (3, 64, 64)
    save_multimodal_checkpoint(model, tmp_path, object(), {"positive_class": 1})
    loaded, _, config = load_multimodal_checkpoint(tmp_path)
    assert config["embedding_dim"] == 16
    assert loaded(clinical, ecg, image)[0].shape == (3, 2)


def test_multimodal_training_fixture(tmp_path):
    rows = 30
    clinical = pd.DataFrame({"patient_id": [f"p{i}" for i in range(rows)],
                             "target": [i % 2 for i in range(rows)],
                             "age": [35 + i for i in range(rows)],
                             "sex": ["F" if i % 3 else "M" for i in range(rows)]})
    ecg_rows, image_rows = [], []
    ecg_dir, image_dir = tmp_path / "ecg", tmp_path / "images"
    ecg_dir.mkdir(); image_dir.mkdir()
    for i in range(rows):
        ecg_path = ecg_dir / f"p{i}.csv"
        np.savetxt(ecg_path, np.sin(np.linspace(0, 10, 256) + i), delimiter=",")
        image_path = image_dir / f"p{i}.png"
        Image.fromarray(np.full((32, 32, 3), i * 7, dtype=np.uint8)).save(image_path)
        ecg_rows.append({"patient_id": f"p{i}", "ecg_path": ecg_path.name})
        image_rows.append({"patient_id": f"p{i}", "image_path": image_path.name})
    clinical_path = tmp_path / "clinical.csv"
    ecg_manifest_path = ecg_dir / "manifest.csv"
    image_manifest_path = image_dir / "manifest.csv"
    clinical.to_csv(clinical_path, index=False)
    pd.DataFrame(ecg_rows).to_csv(ecg_manifest_path, index=False)
    pd.DataFrame(image_rows).to_csv(image_manifest_path, index=False)
    metrics = train_multimodal(SimpleNamespace(
        clinical=str(clinical_path), ecg_manifest=str(ecg_manifest_path),
        image_manifest=str(image_manifest_path), patient_id="patient_id", target="target",
        positive_label=None, output=str(tmp_path / "artifact"), epochs=1, batch_size=4,
        embedding_dim=8, ecg_samples=256, image_size=32, learning_rate=1e-3,
        contrastive_weight=0.1))
    assert "test" in metrics and "clinical_logistic_test" in metrics
    assert (tmp_path / "artifact" / "multimodal_weights.pt").exists()
    subgroup_report = evaluate_multimodal_subgroups(
        clinical_path, ecg_manifest_path, image_manifest_path, "target", ["sex"],
        tmp_path / "artifact", min_group_size=1)
    assert set(subgroup_report) == {"sex"}
    assert all("status" in report or "roc_auc" in report for report in subgroup_report["sex"].values())

    # Separate temporary patients exercise the frozen-model external evaluation path.
    external = tmp_path / "external"
    external_ecg, external_images = external / "ecg", external / "images"
    external_ecg.mkdir(parents=True); external_images.mkdir()
    external_clinical_rows, external_ecg_rows, external_image_rows = [], [], []
    for i in range(20):
        pid = f"external_{i}"
        external_clinical_rows.append({"patient_id": pid, "target": i % 2,
                                       "age": 50 + i, "sex": "F" if i % 2 else "M"})
        signal_path = external_ecg / f"{pid}.csv"
        np.savetxt(signal_path, np.cos(np.linspace(0, 9, 256) + i), delimiter=",")
        image_path = external_images / f"{pid}.png"
        Image.fromarray(np.full((32, 32, 3), i * 5, dtype=np.uint8)).save(image_path)
        external_ecg_rows.append({"patient_id": pid, "ecg_path": signal_path.name})
        external_image_rows.append({"patient_id": pid, "image_path": image_path.name})
    external_clinical_path = external / "clinical.csv"
    external_ecg_manifest = external_ecg / "manifest.csv"
    external_image_manifest = external_images / "manifest.csv"
    pd.DataFrame(external_clinical_rows).to_csv(external_clinical_path, index=False)
    pd.DataFrame(external_ecg_rows).to_csv(external_ecg_manifest, index=False)
    pd.DataFrame(external_image_rows).to_csv(external_image_manifest, index=False)
    model, _config, alignment, c, e, im, labels = load_paired_evaluation(
        external_clinical_path, external_ecg_manifest, external_image_manifest,
        "target", tmp_path / "artifact", require_external_independence=True)
    external_metrics = evaluate_logits(model, c, e, im, labels)
    assert alignment["complete_triplets"] == 20
    assert "pr_auc" in external_metrics and "specificity" in external_metrics
    noisy = evaluate_logits(model, c, e + 0.02, im, labels)
    missing_ecg = evaluate_logits(model, c, None, im, labels)
    assert noisy["n"] == missing_ecg["n"] == 20

    # External evaluation rejects even a single patient reused from training.
    overlapping = pd.read_csv(external_clinical_path)
    overlapping.loc[0, "patient_id"] = "p0"
    overlapping.to_csv(external_clinical_path, index=False)
    with pytest.raises(ValueError, match="used for training"):
        load_paired_evaluation(external_clinical_path, external_ecg_manifest,
                               external_image_manifest, "target", tmp_path / "artifact",
                               require_external_independence=True)
