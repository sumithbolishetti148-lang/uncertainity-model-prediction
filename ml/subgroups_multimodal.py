"""Evaluate a trained multimodal checkpoint across groups in its held-out test patients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.multimodal_evaluation import evaluate_logits, load_paired_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def evaluate_multimodal_subgroups(clinical_path, ecg_manifest_path, image_manifest_path,
                                 target, group_columns, model_dir=PROJECT_ROOT / "models" / "multimodal",
                                 min_group_size=10):
    model, config, alignment, clinical, ecg, image, labels = load_paired_evaluation(
        clinical_path, ecg_manifest_path, image_manifest_path, target, model_dir,
        patient_ids=config_test_ids(model_dir))
    frame = pd.read_csv(clinical_path)
    missing = sorted(set(group_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Clinical data is missing subgroup column(s): {missing}")
    id_column = config["patient_id_column"]
    groups_by_id = frame.set_index(frame[id_column].astype(str))[list(group_columns)]
    test_ids = alignment["evaluation_patient_ids"]
    result = {}
    for column in group_columns:
        result[column] = {}
        for value, rows in groups_by_id.loc[test_ids].groupby(column, dropna=True):
            key = str(value)
            patient_ids = rows.index.astype(str).tolist()
            indices = [i for i, patient_id in enumerate(test_ids) if patient_id in set(patient_ids)]
            subgroup_labels = labels[indices]
            if len(indices) < min_group_size or len(set(subgroup_labels.tolist())) != 2:
                result[column][key] = {
                    "status": "insufficient_data", "n": len(indices),
                    "minimum": min_group_size,
                    "reason": "requires minimum held-out patients and both target classes",
                }
                continue
            result[column][key] = evaluate_logits(
                model, clinical[indices], ecg[indices], image[indices], subgroup_labels)
    return result


def config_test_ids(model_dir):
    """Read held-out IDs early so the shared loader evaluates only the internal test split."""
    from ml.multimodal import load_multimodal_checkpoint

    _model, _preprocessor, config = load_multimodal_checkpoint(model_dir)
    test_ids = config.get("internal_test_patient_ids")
    if not test_ids:
        raise ValueError("Checkpoint lacks internal test patient IDs; retrain with the current pipeline")
    return test_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--ecg-manifest", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-columns", nargs="+", required=True)
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models" / "multimodal"))
    parser.add_argument("--min-group-size", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate_multimodal_subgroups(
        args.clinical, args.ecg_manifest, args.image_manifest, args.target,
        args.group_columns, args.model_dir, args.min_group_size), indent=2))
