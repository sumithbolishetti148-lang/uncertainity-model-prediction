from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from ml.pipeline import ARTIFACT, load_model, predict_one
from ml.multimodal import (gradcam_1d, gradcam_2d, load_multimodal_checkpoint,
                           mc_dropout_predict)

def _project_path(variable: str, default: Path) -> Path:
    configured = Path(os.environ.get(variable, str(default))).expanduser()
    return configured.resolve() if configured.is_absolute() else (PROJECT_ROOT / configured).resolve()


MODEL_PATH = _project_path("CVD_MODEL_PATH", ARTIFACT)
MULTIMODAL_PATH = _project_path("CVD_MULTIMODAL_MODEL_DIR", PROJECT_ROOT / "models" / "multimodal")
app = FastAPI(title="CVD Research Prediction API", version="1.0.0")


def _json_safe(value):
    """Convert NumPy scalar/array values in saved reports to standard JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (ValueError, TypeError):
            pass
    return value


def load_clinical_artifact():
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        baseline_path = MULTIMODAL_PATH / "clinical_baseline.joblib"
        if baseline_path.is_file():
            import joblib
            return joblib.load(baseline_path)
        raise


class PredictionRequest(BaseModel):
    clinical: dict[str, Any]


class MultimodalRequest(BaseModel):
    clinical: dict[str, Any]
    ecg: list[float] | None = None
    # Numeric RGB data as HxWx3 or 3xHxW, normalized to [0, 1].
    image: list[Any] | None = None
    passes: int = Field(default=30, ge=2, le=100)


def _multimodal_inputs(request: MultimodalRequest):
    import numpy as np
    import pandas as pd
    import torch

    try:
        model, preprocessor, config = load_multimodal_checkpoint(MULTIMODAL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Multimodal model has not been trained; see README.") from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=503, detail=f"Could not load multimodal model: {exc}") from exc
    feature_names = list(preprocessor.feature_names_in_)
    missing = [name for name in feature_names if name not in request.clinical]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing clinical feature(s): {missing}")
    try:
        clinical = preprocessor.transform(pd.DataFrame([{k: request.clinical[k] for k in feature_names}]))
        if hasattr(clinical, "toarray"):
            clinical = clinical.toarray()
        inputs = {"clinical": torch.tensor(np.asarray(clinical, dtype=np.float32))}
        if request.ecg is not None:
            signal = np.asarray(request.ecg, dtype=np.float32)
            if signal.ndim != 1 or len(signal) < 16 or not np.isfinite(signal).all():
                raise ValueError("ECG must be a finite one-dimensional numeric signal with at least 16 samples")
            sd = float(signal.std())
            if sd > 1e-8:
                signal = (signal - signal.mean()) / sd
            inputs["ecg"] = torch.tensor(signal[None, :])
        if request.image is not None:
            image = np.asarray(request.image, dtype=np.float32)
            if image.ndim != 3:
                raise ValueError("Image must be a three-dimensional RGB array")
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            if image.shape[0] != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Image must have 3 RGB channels and finite values in [0, 1]")
            inputs["image"] = torch.tensor(image[None, ...])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid multimodal input: {exc}") from exc
    return model, config, inputs


async def _clinical_upload(file: UploadFile):
    import io
    import pandas as pd
    if Path(file.filename or "").suffix.lower() != ".csv":
        raise HTTPException(status_code=422, detail="Upload a CSV file")
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV exceeds the 20 MB limit")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=422, detail="CSV contains no rows")
    return frame


def _evaluate_frame(frame, target, artifact):
    from ml.pipeline import _metrics
    features = artifact["features"]
    if target not in frame:
        raise HTTPException(status_code=422, detail=f"Target column '{target}' is missing")
    missing = [f for f in features if f not in frame]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model feature(s): {missing}")
    frame = frame.dropna(subset=[target])
    if frame[target].nunique() != 2:
        raise HTTPException(status_code=422, detail="Evaluation CSV target must contain both trained classes")
    return _metrics(artifact["model"], frame[features], frame[target], artifact["positive_class"])


@app.get("/health")
def health():
    multimodal_loaded = all((MULTIMODAL_PATH / name).is_file() for name in (
        "model_config.json", "multimodal_weights.pt", "clinical_preprocessor.joblib"))
    clinical_loaded = MODEL_PATH.is_file() or (MULTIMODAL_PATH / "clinical_baseline.joblib").is_file()
    return {"status": "ok", "model_loaded": clinical_loaded or multimodal_loaded,
            "clinical_model_loaded": clinical_loaded,
            "multimodal_model_loaded": multimodal_loaded,
            "modality": "multimodal" if multimodal_loaded else "clinical_only"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_one(request.clinical, load_clinical_artifact())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/metadata")
def metadata():
    try:
        multi_exists = all((MULTIMODAL_PATH / name).is_file() for name in (
            "model_config.json", "multimodal_weights.pt", "clinical_preprocessor.joblib"))
        try:
            artifact = load_model(MODEL_PATH)
            payload = {"features": artifact["features"], "classes": artifact["classes"],
                       "numeric_features": artifact["numeric_features"], "metrics": artifact["metrics"],
                       "modality": artifact["modality"]}
        except FileNotFoundError:
            baseline_path = MULTIMODAL_PATH / "clinical_baseline.joblib"
            if baseline_path.is_file():
                import joblib
                artifact = joblib.load(baseline_path)
                pipeline = artifact["model"]
                transformers = pipeline.named_steps["preprocess"].transformers
                numeric = next((list(cols) for name, _transformer, cols in transformers if name == "numeric"), [])
                payload = {"features": artifact["features"],
                           "classes": list(pipeline.named_steps["model"].classes_),
                           "numeric_features": numeric, "metrics": {"test": artifact["metrics"]},
                           "modality": "clinical_only"}
            elif multi_exists:
                _model, preprocessor, config = load_multimodal_checkpoint(MULTIMODAL_PATH)
                transformers = preprocessor.transformers
                numeric = next((list(cols) for name, _transformer, cols in transformers if name == "numeric"), [])
                payload = {"features": list(preprocessor.feature_names_in_), "classes": config["classes"],
                           "numeric_features": numeric, "metrics": config.get("metrics", {}),
                           "modality": "multimodal_trained"}
            else:
                raise FileNotFoundError(
                    f"No trained model is available. Run the training command first (see README). "
                    f"Checked {MODEL_PATH} and {MULTIMODAL_PATH}.")
        payload["multimodal_model_loaded"] = multi_exists
        return _json_safe(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/multimodal")
def predict_multimodal(request: MultimodalRequest):
    import torch
    model, config, inputs = _multimodal_inputs(request)
    try:
        result = mc_dropout_predict(model, inputs, passes=request.passes)
        positive_index = 1
        probability = float(result["mean_probability"][0, positive_index])
        uncertainty = float(result["variance"][0])
        available = [name for name in ("clinical", "ecg", "image") if name in inputs]
        return {"prediction": "CVD Detected" if probability >= 0.5 else "No CVD Detected",
                "probability": probability, "uncertainty": uncertainty,
                "uncertainty_method": result["method"], "uncertainty_level":
                "Low" if uncertainty < 0.002 else "Medium" if uncertainty < 0.02 else "High",
                "available_modalities": available,
                "missing_modalities": [m for m in ("clinical", "ecg", "image") if m not in inputs],
                "note": "Predictive variance from MC Dropout; not medical certainty.",
                "target_classes": config["classes"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multimodal prediction failed: {exc}") from exc


@app.post("/explain/gradcam")
def explain_gradcam(request: MultimodalRequest):
    import numpy as np
    import torch
    model, _config, inputs = _multimodal_inputs(request)
    try:
        if "image" in inputs:
            heatmap = gradcam_2d(model, inputs["image"], clinical=inputs["clinical"], ecg=inputs.get("ecg"))
            modality = "image"
        elif "ecg" in inputs:
            heatmap = gradcam_1d(model, inputs["ecg"], clinical=inputs["clinical"])
            modality = "ecg"
        else:
            raise HTTPException(status_code=422, detail="Grad-CAM requires an ECG signal or image input")
        return {"modality": modality, "method": "prediction-targeted Grad-CAM",
                "heatmap": heatmap.detach().cpu().numpy().tolist()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}") from exc


@app.post("/validate/external")
async def validate_external(file: UploadFile = File(...), target: str = Form(...)):
    artifact = load_clinical_artifact()
    frame = await _clinical_upload(file)
    training_ids = artifact.get("training_patient_ids")
    if training_ids is not None:
        id_column = next((c for c in frame.columns if c.lower() in {"patient_id", "patientid", "subject_id"}), None)
        if id_column is None:
            raise HTTPException(status_code=422, detail="External CSV must include patient IDs to verify independence from training data")
        overlap = set(map(str, training_ids)) & set(frame[id_column].dropna().astype(str))
        if overlap:
            raise HTTPException(status_code=422, detail=f"External CSV contains patient IDs used for training: {sorted(overlap)[:8]}")
    return {"dataset_type": "external", "training_used": False,
            "metrics": _evaluate_frame(frame, target, artifact)}


@app.post("/evaluate/robustness")
async def evaluate_robustness(file: UploadFile = File(...), target: str = Form(...)):
    import numpy as np
    from ml.pipeline import _metrics
    artifact = load_clinical_artifact()
    frame = await _clinical_upload(file)
    base = _evaluate_frame(frame, target, artifact)
    features = artifact["features"]
    y = frame[target]
    missing = frame[features].copy()
    numeric = artifact.get("numeric_features", [])
    if numeric:
        missing.loc[:, numeric[0]] = np.nan
    missing_metrics = _metrics(artifact["model"], missing, y, artifact["positive_class"])
    perturbed = frame[features].copy()
    rng = np.random.default_rng(42)
    for col in numeric:
        scale = perturbed[col].std(skipna=True)
        if np.isfinite(scale) and scale > 0:
            perturbed[col] = perturbed[col] + rng.normal(0, 0.05 * scale, len(perturbed))
    noise_metrics = _metrics(artifact["model"], perturbed, y, artifact["positive_class"])
    return {"method": "clinical missing-feature and 0.05-SD perturbation checks",
            "baseline": base, "first_numeric_feature_missing": missing_metrics,
            "numeric_perturbation_0.05_sd": noise_metrics}


@app.post("/evaluate/subgroups")
async def evaluate_subgroups(file: UploadFile = File(...), target: str = Form(...),
                             group_columns: str = Form(...), min_group_size: int = Form(10)):
    from ml.pipeline import _metrics
    artifact = load_clinical_artifact()
    frame = await _clinical_upload(file)
    features = artifact["features"]
    missing = [f for f in features if f not in frame]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model feature(s): {missing}")
    columns = [col.strip() for col in group_columns.split(",") if col.strip()]
    if not columns or target not in frame or any(col not in frame for col in columns):
        raise HTTPException(status_code=422, detail="Provide an existing target and comma-separated subgroup columns")
    report = {}
    for column in columns:
        report[column] = {}
        for value, group in frame.dropna(subset=[column, target]).groupby(column):
            if len(group) < min_group_size or group[target].nunique() != 2:
                report[column][str(value)] = {"status": "insufficient_data", "n": int(len(group))}
            else:
                report[column][str(value)] = _metrics(artifact["model"], group[features],
                                                       group[target], artifact["positive_class"])
    return {"model": "clinical_baseline", "subgroups": report}
