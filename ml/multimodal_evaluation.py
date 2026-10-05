"""Shared paired-data evaluation helpers for the trained neural checkpoint."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.multimodal import load_multimodal_checkpoint
from ml.multimodal_data import align_patient_modalities, load_ecg, load_medical_image
from ml.train_multimodal import classification_metrics


def load_paired_evaluation(clinical_path, ecg_manifest_path, image_manifest_path, target, model_dir,
                           *, require_external_independence=False, patient_ids=None):
    model, preprocessor, config = load_multimodal_checkpoint(model_dir)
    clinical = pd.read_csv(clinical_path)
    aligned, alignment = align_patient_modalities(
        clinical, pd.read_csv(ecg_manifest_path), pd.read_csv(image_manifest_path),
        clinical_id=config["patient_id_column"],
        ecg_manifest_dir=Path(ecg_manifest_path).parent,
        image_manifest_dir=Path(image_manifest_path).parent)
    if require_external_independence:
        training_ids = config.get("training_patient_ids")
        if not training_ids:
            raise ValueError("Checkpoint lacks training patient IDs; independent external validation cannot be verified. Retrain with the current pipeline.")
        overlap = set(map(str, training_ids)) & set(aligned.patient_id.astype(str))
        if overlap:
            sample = sorted(overlap)[:8]
            raise ValueError(f"External dataset contains patient IDs used for training: {sample}")
        alignment["training_patient_overlap"] = 0
    if target not in aligned:
        raise ValueError(f"Target column '{target}' missing from evaluation clinical file")
    complete = aligned.dropna(subset=[target]).loc[lambda d: d.has_ecg & d.has_image].reset_index(drop=True)
    if patient_ids is not None:
        requested_ids = set(map(str, patient_ids))
        complete = complete.loc[complete.patient_id.astype(str).isin(requested_ids)].reset_index(drop=True)
        found_ids = set(complete.patient_id.astype(str))
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Requested evaluation patient IDs are missing complete aligned records: {missing_ids[:8]}")
    if complete.empty:
        raise ValueError("Evaluation has no complete patient-ID-aligned triplets")
    alignment["evaluation_patient_ids"] = complete.patient_id.astype(str).tolist()
    X = complete.drop(columns=[target, "patient_id", "ecg_path", "image_path", "has_ecg", "has_image"])
    clinical_x = preprocessor.transform(X)
    if hasattr(clinical_x, "toarray"):
        clinical_x = clinical_x.toarray()
    clinical_x = torch.tensor(np.asarray(clinical_x, dtype=np.float32))
    ecg_x = torch.tensor(np.stack([load_ecg(p) for p in complete.ecg_path]), dtype=torch.float32)
    image_x = torch.tensor(np.stack([load_medical_image(p) for p in complete.image_path]), dtype=torch.float32)
    y = (complete[target].astype(str).to_numpy() == str(config["positive_class"])).astype(np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("Evaluation set must contain both positive and negative target classes")
    return model, config, alignment, clinical_x, ecg_x, image_x, y


def evaluate_logits(model, clinical, ecg, image, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(clinical=clinical, ecg=ecg, image=image)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return classification_metrics(y, probabilities)
