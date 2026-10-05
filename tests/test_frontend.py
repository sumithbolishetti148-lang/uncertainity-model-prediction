from types import SimpleNamespace
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.app import main as api
from ml.pipeline import train


class LocalResponse:
    def __init__(self, response):
        self.response = response
        self.ok = response.is_success

    def json(self):
        return self.response.json()


def test_frontend_prediction_and_remove_upload(tmp_path, monkeypatch):
    frame = pd.DataFrame({"age": range(40, 160), "risk": [i % 2 for i in range(120)]})
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    model_path = tmp_path / "model.joblib"
    train(data, "risk", model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    client = TestClient(api.app)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: LocalResponse(client.get("/metadata")))
    monkeypatch.setattr(requests, "post", lambda *_a, **kw: LocalResponse(client.post("/predict", json=kw["json"])))

    ui = AppTest.from_file("frontend/app.py", default_timeout=60).run()
    assert len(ui.exception) == 0


    assert len(ui.number_input) == 1
    ui.number_input[0].set_value(70)
    ui.button[0].click().run()
    assert len(ui.exception) == 0
    assert any("CVD Detected" in item.value or "No CVD Detected" in item.value
               for item in ui.get("markdown"))

    # AppTest does not expose a file chooser setter; seed the widget's upload value
    # to exercise the visible selected-file and Remove File interaction.
    ecg_csv = "\n".join(f"{i},{i + 1}" for i in range(20)).encode()
    ui.session_state["ecg_upload_0"] = SimpleNamespace(
        name="trace.csv", size=len(ecg_csv), getvalue=lambda: ecg_csv)
    ui.run()
    assert any(button.label == "Remove File" for button in ui.button)
    next(button for button in ui.button if button.label == "Remove File").click().run()
    assert not any(button.label == "Remove File" for button in ui.button)
    assert len(ui.exception) == 0

    image_bytes = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(image_bytes, format="PNG")
    data = image_bytes.getvalue()
    ui.session_state["medical_image_0"] = SimpleNamespace(
        name="image.png", size=len(data), getvalue=lambda: data)
    ui.run()
    assert any(button.label == "Remove Image" for button in ui.button)
    next(button for button in ui.button if button.label == "Remove Image").click().run()
    assert not any(button.label == "Remove Image" for button in ui.button)
    assert len(ui.exception) == 0


def test_frontend_stays_usable_when_no_model_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing-clinical.joblib")
    monkeypatch.setattr(api, "MULTIMODAL_PATH", tmp_path / "missing-multimodal")
    client = TestClient(api.app)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: LocalResponse(client.get("/metadata")))

    ui = AppTest.from_file("frontend/app.py", default_timeout=60).run()
    assert len(ui.exception) == 0
    assert any("No trained model is available" in warning.value for warning in ui.warning)
    subheaders = [item.value for item in ui.subheader]
    assert "Optional ECG file" in subheaders
    assert "Optional medical image" in subheaders
