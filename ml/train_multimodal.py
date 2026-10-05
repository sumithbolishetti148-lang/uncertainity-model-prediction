"""Train paired clinical+ECG+image model from ID-keyed manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.multimodal import (MultimodalCVDModel, UnimodalCVDClassifier,
                           cross_modal_infonce, save_multimodal_checkpoint)
from ml.multimodal_data import align_patient_modalities, load_ecg, load_medical_image
from ml.pipeline import _positive_label, make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PairedDataset(Dataset):
    def __init__(self, clinical, ecg, images, labels):
        self.clinical = torch.tensor(np.asarray(clinical, dtype=np.float32))
        self.ecg = torch.tensor(np.stack(ecg), dtype=torch.float32)
        self.images = torch.tensor(np.stack(images), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.clinical[index], self.ecg[index], self.images[index], self.labels[index], index


def classification_metrics(y, probability, positive_index=1):
    predicted = probability.argmax(axis=1)
    truth = np.asarray(y)
    positive = truth == positive_index
    pred_positive = predicted == positive_index
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[1-positive_index, positive_index]).ravel()
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision_score(positive, pred_positive, zero_division=0)),
            "recall_sensitivity": float(recall_score(positive, pred_positive, zero_division=0)),
            "specificity": float(tn / max(1, tn + fp)),
            "f1": float(f1_score(positive, pred_positive, zero_division=0)),
            "roc_auc": float(roc_auc_score(positive, probability[:, positive_index])),
            "pr_auc": float(average_precision_score(positive, probability[:, positive_index])),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "confusion_matrix_labels": ["negative", "positive"], "n": int(len(truth))}


def train(args):
    clinical = pd.read_csv(args.clinical)
    ecg_manifest = pd.read_csv(args.ecg_manifest)
    image_manifest = pd.read_csv(args.image_manifest)
    aligned, alignment = align_patient_modalities(
        clinical, ecg_manifest, image_manifest, clinical_id=args.patient_id,
        ecg_manifest_dir=Path(args.ecg_manifest).parent,
        image_manifest_dir=Path(args.image_manifest).parent)
    print("Alignment report:", json.dumps(alignment, indent=2))
    if args.target not in aligned:
        raise ValueError(f"Target column '{args.target}' not found in the clinical table")
    complete = aligned.dropna(subset=[args.target]).loc[lambda d: d.has_ecg & d.has_image].reset_index(drop=True)
    if complete.empty:
        raise ValueError("No patients have complete, ID-matched clinical, ECG, and image records")
    class_counts = complete[args.target].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 7:
        raise ValueError("Need at least 7 complete patients in each target class for a stratified 70/15/15 split")
    ids = complete.patient_id.astype(str).to_numpy()
    labels = complete[args.target].to_numpy()
    indices = np.arange(len(complete))
    train_rest, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_rest, test_size=0.15/0.85, random_state=42,
                                          stratify=labels[train_rest])
    positive = _positive_label(list(pd.unique(labels)), args.positive_label)
    class_values = [c for c in pd.unique(labels) if c != positive] + [positive]
    y = np.asarray([class_values.index(value) for value in labels], dtype=np.int64)
    X = complete.drop(columns=[args.target, "patient_id", "ecg_path", "image_path", "has_ecg", "has_image"])
    preprocessor = clone(make_pipeline(X.iloc[train_idx]).named_steps["preprocess"])
    clinical_train = preprocessor.fit_transform(X.iloc[train_idx])
    clinical_val = preprocessor.transform(X.iloc[val_idx])
    clinical_test = preprocessor.transform(X.iloc[test_idx])
    if hasattr(clinical_train, "toarray"):
        clinical_train, clinical_val, clinical_test = [x.toarray() for x in (clinical_train, clinical_val, clinical_test)]
    clinical_all = np.zeros((len(complete), clinical_train.shape[1]), dtype=np.float32)
    for part_idx, part in ((train_idx, clinical_train), (val_idx, clinical_val), (test_idx, clinical_test)):
        clinical_all[part_idx] = part

    ecgs = [load_ecg(path, samples=args.ecg_samples) for path in complete.ecg_path]
    images = [load_medical_image(path, size=args.image_size) for path in complete.image_path]
    dataset = PairedDataset(clinical_all, ecgs, images, y)
    loader = DataLoader(torch.utils.data.Subset(dataset, train_idx.tolist()), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = MultimodalCVDModel(clinical_all.shape[1], embedding_dim=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_counts = np.bincount(y[train_idx], minlength=2)
    class_weights = torch.tensor(len(train_idx) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for c, e, im, target, local_idx in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(c, e, im)
            ce = criterion(logits, target)
            batch_ids = [ids[i] for i in local_idx.tolist()]
            contrastive = (cross_modal_infonce(embeddings, batch_ids)
                           if len(batch_ids) >= 2 else ce.new_zeros(()))
            loss = ce + args.contrastive_weight * contrastive
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.4f}")

    model.eval()
    outputs = {}
    with torch.no_grad():
        for name, idx in (("validation", val_idx), ("test", test_idx)):
            logits, _ = model(dataset.clinical[idx], dataset.ecg[idx], dataset.images[idx])
            outputs[name] = classification_metrics(y[idx], torch.softmax(logits, dim=1).numpy())

    unimodal_reports = {}
    all_inputs = {"clinical": dataset.clinical, "ecg": dataset.ecg, "image": dataset.images}
    for modality, tensor in all_inputs.items():
        torch.manual_seed(42)
        branch = UnimodalCVDClassifier(modality, clinical_all.shape[1], args.embedding_dim)
        branch_optimizer = torch.optim.Adam(branch.parameters(), lr=args.learning_rate)
        branch.train()
        for _epoch in range(args.epochs):
            for c, e, im, target, _idx in loader:
                branch_optimizer.zero_grad(set_to_none=True)
                batch_input = {"clinical": c, "ecg": e, "image": im}[modality]
                branch_loss = criterion(branch(batch_input), target)
                branch_loss.backward()
                branch_optimizer.step()
        branch.eval()
        with torch.no_grad():
            val_probs = torch.softmax(branch(tensor[val_idx]), dim=1).numpy()
            test_probs = torch.softmax(branch(tensor[test_idx]), dim=1).numpy()
        unimodal_reports[modality] = {
            "validation": classification_metrics(y[val_idx], val_probs),
            "test": classification_metrics(y[test_idx], test_probs)}
        branch_dir = Path(args.output) / f"unimodal_{modality}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(branch.state_dict(), branch_dir / "weights.pt")
        (branch_dir / "config.json").write_text(json.dumps({
            "modality": modality, "clinical_dim": clinical_all.shape[1],
            "embedding_dim": args.embedding_dim, "classes": [str(v) for v in class_values]}, indent=2))
    # Train and evaluate the preserved clinical logistic baseline on the exact same split.
    baseline = make_pipeline(X.iloc[train_idx])
    baseline.fit(X.iloc[train_idx], labels[train_idx])
    baseline_classes = list(baseline.named_steps["model"].classes_)
    positive_prob = baseline.predict_proba(X.iloc[test_idx])[:, baseline_classes.index(positive)]
    baseline_pred = baseline.predict(X.iloc[test_idx])
    baseline_y = labels[test_idx] == positive
    tn, fp, fn, tp = confusion_matrix(baseline_y, baseline_pred == positive, labels=[False, True]).ravel()
    outputs["clinical_logistic_test"] = {
        "accuracy": float(accuracy_score(baseline_y, baseline_pred == positive)),
        "precision": float(precision_score(baseline_y, baseline_pred == positive, zero_division=0)),
        "recall_sensitivity": float(recall_score(baseline_y, baseline_pred == positive, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(baseline_y, baseline_pred == positive, zero_division=0)),
        "roc_auc": float(roc_auc_score(baseline_y, positive_prob)),
        "pr_auc": float(average_precision_score(baseline_y, positive_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "n": int(len(test_idx))}
    outputs["unimodal_test"] = unimodal_reports
    output = Path(args.output)
    save_multimodal_checkpoint(model, output, preprocessor,
                               {"patient_id_column": "patient_id", "target": args.target,
                               "training_patient_ids": ids.tolist(),
                                "internal_test_patient_ids": ids[test_idx].tolist(),
                                "classes": [str(x) for x in class_values], "positive_class": str(positive),
                                "modality_order": ["clinical", "ecg", "image"], "seed": 42,
                                "alignment": alignment, "metrics": outputs,
                                "loss": "cross_entropy + contrastive_weight * symmetric InfoNCE"})
    joblib.dump({"model": baseline, "features": X.columns.tolist(), "positive_class": positive,
                 "classes": list(baseline.named_steps["model"].classes_),
                 "training_patient_ids": ids.tolist(),
                 "numeric_features": X.select_dtypes(include=["number", "bool"]).columns.tolist(),
                 "shap_background": X.iloc[train_idx].sample(n=min(100, len(train_idx)), random_state=42),
                 "metrics": outputs["clinical_logistic_test"]}, output / "clinical_baseline.joblib")
    print(json.dumps(outputs, indent=2))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True, help="CSV with one row/patient, target, and actual clinical features")
    parser.add_argument("--patient-id", default="patient_id")
    parser.add_argument("--target", required=True)
    parser.add_argument("--ecg-manifest", required=True, help="CSV columns patient_id,ecg_path (or ecg_file); paths relative to manifest")
    parser.add_argument("--image-manifest", required=True, help="CSV columns patient_id,image_path (or image_file); paths relative to manifest")
    parser.add_argument("--positive-label")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "multimodal"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ecg-samples", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    train(parser.parse_args())
