import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app import main as api
from ml import pipeline
from ml.multimodal import MultimodalCVDModel, save_multimodal_checkpoint


def test_api_health_and_prediction(tmp_path, monkeypatch):
    frame = pd.DataFrame({"age": range(40, 160), "risk": [i % 2 for i in range(120)]})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    artifact_path = tmp_path / "model.joblib"
    pipeline.train(path, "risk", artifact_path)
    monkeypatch.setattr(pipeline, "ARTIFACT", artifact_path)
    monkeypatch.setattr("backend.app.main.MODEL_PATH", artifact_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    metadata = client.get("/metadata")
    assert metadata.status_code == 200, metadata.text
    response = client.post("/predict", json={"clinical": {"age": 70}})
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1
    assert client.post("/predict", json={"clinical": {}}).status_code == 422


def test_multimodal_prediction_gradcam_and_validation(tmp_path, monkeypatch):
    frame = pd.DataFrame({"age": range(40, 160), "risk": [i % 2 for i in range(120)]})
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    artifact_path = tmp_path / "clinical.joblib"
    artifact = pipeline.train(data, "risk", artifact_path)
    prep = artifact["model"].named_steps["preprocess"]
    dim = prep.transform(frame[["age"]]).shape[1]
    multi_path = tmp_path / "multimodal"
    model = MultimodalCVDModel(dim, embedding_dim=8)
    save_multimodal_checkpoint(model, multi_path, prep,
                               {"classes": ["0", "1"], "positive_class": "1"})
    monkeypatch.setattr("backend.app.main.MULTIMODAL_PATH", multi_path)
    client = TestClient(app)
    request = {"clinical": {"age": 70}, "ecg": [float(i % 4) for i in range(64)], "passes": 4}
    prediction = client.post("/predict/multimodal", json=request)
    assert prediction.status_code == 200, prediction.text
    assert 0 <= prediction.json()["probability"] <= 1
    assert prediction.json()["available_modalities"] == ["clinical", "ecg"]
    explanation = client.post("/explain/gradcam", json=request)
    assert explanation.status_code == 200, explanation.text
    assert len(explanation.json()["heatmap"][0]) == 64
    invalid = client.post("/predict/multimodal", json={"clinical": {"age": 70}, "ecg": [1.0]})
    assert invalid.status_code == 422


def test_missing_multimodal_model_is_service_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.main.MULTIMODAL_PATH", tmp_path / "absent")
    response = TestClient(app).post("/predict/multimodal", json={"clinical": {"age": 70}})
    assert response.status_code == 503


def test_relative_model_paths_are_resolved_from_project_root(monkeypatch):
    monkeypatch.setenv("CVD_MODEL_PATH", "models/example.joblib")
    assert api._project_path("CVD_MODEL_PATH", api.ARTIFACT) == api.PROJECT_ROOT / "models" / "example.joblib"
