"""Leakage-safe tabular training pipeline; all preprocessing fits on train only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "clinical_model.joblib"


def load_dataset(path: str | Path, target: str, *, include_groups: bool = False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Dataset has no rows")
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(frame.columns)}")
    frame = frame.dropna(subset=[target]).copy()
    if frame[target].nunique() != 2:
        raise ValueError("Target must contain exactly two non-missing classes")
    patient = next((c for c in frame.columns if c.lower() in {"patient_id", "patientid", "subject_id"}), None)
    y = frame.pop(target)
    groups = frame[patient].copy() if patient else None
    X = frame.drop(columns=[patient]) if patient else frame
    if X.shape[1] == 0:
        raise ValueError("No clinical features remain after excluding target/patient ID")
    invalid_numeric = int(np.isinf(X.select_dtypes(include=["number"]).to_numpy(dtype=float)).sum())
    if invalid_numeric:
        X = X.replace([np.inf, -np.inf], np.nan)
    print(f"Invalid numeric infinities converted to missing values: {invalid_numeric}")
    print(f"Loaded {path}: {len(frame)} rows x {frame.shape[1]} columns; after target filtering: {len(X)} rows")
    print("Class counts:", y.value_counts(dropna=False).to_dict())
    if include_groups:
        return X, y, groups
    return X, y


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    # Balanced weights are determined only from the training labels during fit.
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED
    ))])


def _positive_label(classes: list[Any], requested: Any = None) -> Any:
    if requested is not None:
        for candidate in classes:
            if requested == candidate or str(requested) == str(candidate):
                return candidate
            try:
                if float(requested) == float(candidate):
                    return candidate
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Positive label {requested!r} not found in target classes {classes}")
    if set(classes) == {0, 1} or set(classes) == {False, True}:
        return next(c for c in classes if c in (1, True))
    positive_names = {"1", "true", "yes", "positive", "pos", "case", "cvd", "disease", "present", "high_risk"}
    negative_names = {"0", "false", "no", "negative", "neg", "control", "no_cvd", "no_disease", "absent", "low_risk"}
    candidates = [c for c in classes if str(c).strip().lower().replace(" ", "_") in positive_names]
    if len(candidates) == 1:
        return candidates[0]
    negatives = [c for c in classes if str(c).strip().lower().replace(" ", "_") in negative_names]
    if len(negatives) == 1:
        return next(c for c in classes if c != negatives[0])
    raise ValueError(f"Cannot infer the positive target label from {classes}; pass --positive-label explicitly")


def _metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, positive: Any) -> dict[str, Any]:
    pred = model.predict(X)
    classes = list(model.named_steps["model"].classes_)
    prob = model.predict_proba(X)[:, classes.index(positive)]
    negative = next(c for c in classes if c != positive)
    matrix = confusion_matrix(y, pred, labels=[negative, positive])
    tn, fp, fn, tp = matrix.ravel()
    positive_truth = y == positive
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=positive, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=positive, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=positive, zero_division=0)),
        "roc_auc": float(roc_auc_score(y == positive, prob)),
        "pr_auc": float(average_precision_score(positive_truth, prob)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [str(negative), str(positive)],
        "n": int(len(y)),
    }


def train(path: str | Path, target: str, output: str | Path = ARTIFACT,
          positive_label: Any = None) -> dict[str, Any]:
    X, y, groups = load_dataset(path, target, include_groups=True)
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 7:
        raise ValueError("Need at least 7 labeled rows in each target class for a stratified 70/15/15 split")
    # Split raw records before fitting. Keep all records from a patient together.
    if groups is not None:
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        train_rest, test_idx = next(first.split(X, y, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
        train_rel, val_rel = next(second.split(X.iloc[train_rest], y.iloc[train_rest], groups.iloc[train_rest]))
        train_idx, val_idx = train_rest[train_rel], train_rest[val_rel]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if any(part.nunique() < 2 for part in (y_train, y_val, y_test)):
            raise ValueError("Patient-group splits must each contain both target classes; supply more diverse patients")
    else:
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.30, random_state=SEED, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.50, random_state=SEED, stratify=y_rest)
    print(f"Split rows: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)
    transformed_shape = model.named_steps["preprocess"].transform(X_train).shape
    print(f"Preprocessed training shape: {transformed_shape}")
    classes = model.named_steps["model"].classes_.tolist()
    positive = _positive_label(classes, positive_label)
    report = {"positive_class": positive,
              "validation": _metrics(model, X_val, y_val, positive),
              "test": _metrics(model, X_test, y_test, positive)}
    artifact = {"model": model, "features": X.columns.tolist(), "target": target,
                "classes": classes, "positive_class": positive, "metrics": report,
                "numeric_features": X.select_dtypes(include=["number", "bool"]).columns.tolist(),
                "shap_background": X_train.sample(n=min(100, len(X_train)), random_state=SEED),
                "modality": "clinical_only", "patient_group_split": groups is not None,
                "training_patient_ids": groups.astype(str).tolist() if groups is not None else None,
                "seed": SEED}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    report_path = output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved model: {output}")
    return artifact


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    path = ARTIFACT if path is None else path
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model is missing at {path}; train it first (see README).")
    return joblib.load(path)


def predict_one(values: dict[str, Any], artifact: dict[str, Any], passes: int = 30) -> dict[str, Any]:
    features = artifact["features"]
    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    row = pd.DataFrame([{f: values[f] for f in features}], columns=features)
    classes = list(artifact["model"].named_steps["model"].classes_)
    p = float(artifact["model"].predict_proba(row)[0, classes.index(artifact["positive_class"])])
    negative = next(c for c in artifact["classes"] if c != artifact["positive_class"])
    # Logistic regression is deterministic. Report uncertainty as probability distance
    # from the decision boundary; do not describe it as stochastic/model uncertainty.
    margin = abs(p - 0.5) * 2
    confidence = "High" if margin >= 0.6 else "Medium" if margin >= 0.25 else "Low"
    target_value = str(artifact["positive_class"] if p >= 0.5 else negative)
    return {"prediction": "CVD Detected" if p >= 0.5 else "No CVD Detected",
            "target_value": target_value,
            "probability": p, "confidence": confidence,
            "uncertainty": float(1 - margin), "uncertainty_method": "distance_to_decision_boundary",
            "explanation": explain_one(row, artifact)}


def explain_one(row: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    """Compute SHAP values if available. Never substitute synthetic attributions."""
    try:
        import shap
        model = artifact["model"]
        prep = model.named_steps["preprocess"]
        transformed = prep.transform(row)
        names = prep.get_feature_names_out()
        background = prep.transform(artifact["shap_background"])
        explainer = shap.LinearExplainer(model.named_steps["model"], background)
        vals = np.asarray(explainer.shap_values(transformed)).reshape(-1)
        ordered = sorted(zip(names, vals), key=lambda item: abs(float(item[1])), reverse=True)[:10]
        return {"method": "SHAP LinearExplainer", "top_features": [
            {"feature": str(n).replace("numeric__", "").replace("categorical__", ""), "shap_value": float(v)}
            for n, v in ordered]}
    except Exception as exc:
        return {"method": "unavailable", "top_features": [], "message": f"SHAP could not be computed: {exc}"}


def evaluate_external(path: str | Path, target: str, artifact: dict[str, Any]) -> dict[str, Any]:
    X, y, groups = load_dataset(path, target, include_groups=True)
    training_ids = artifact.get("training_patient_ids")
    if training_ids is not None:
        if groups is None:
            raise ValueError("External dataset must include patient IDs to verify independence from training data")
        overlap = set(map(str, training_ids)) & set(groups.astype(str))
        if overlap:
            raise ValueError(f"External dataset contains patient IDs used for training: {sorted(overlap)[:8]}")
    expected = artifact["features"]
    missing = [f for f in expected if f not in X]
    if missing:
        raise ValueError(f"External dataset lacks model features: {missing}")
    metrics = _metrics(artifact["model"], X[expected], y, artifact["positive_class"])
    metrics["training_patient_overlap"] = 0 if training_ids is not None else "not verifiable: training data had no patient IDs"
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train/evaluate the clinical baseline")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default=str(ARTIFACT))
    parser.add_argument("--positive-label", default=None,
                        help="Positive target value; required if it cannot be inferred (for example, CVD)")
    args = parser.parse_args()
    train(args.data, args.target, args.output, args.positive_label)
