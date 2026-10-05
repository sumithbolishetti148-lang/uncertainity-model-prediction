from io import BytesIO
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from frontend.uploads import read_ecg_upload, validate_image_preview

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def multimodal_payload(clinical, ecg_upload=None, image_upload=None):
    payload = {"clinical": clinical, "passes": 30}
    if ecg_upload:
        signal, _summary = read_ecg_upload(ecg_upload.name, ecg_upload.getvalue())
        payload["ecg"] = signal.tolist()
    if image_upload:
        with Image.open(BytesIO(image_upload.getvalue())) as opened:
            image = np.asarray(opened.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        payload["image"] = image.tolist()
    return payload

st.set_page_config(page_title="CVD Research Predictor", page_icon="❤️", layout="centered")
st.title("Explainable and Uncertainty-Aware Multimodal Deep Learning for Generalizable Cardiovascular Disease Prediction Across Heterogeneous Patient Populations")
st.caption("Research interface for clinical, ECG, and image data. The actual available models depend on supplied data and trained artifacts.")
st.error("Academic and research purposes only. This system is not a medical diagnostic tool. Predictions must not be used for clinical decision-making.")
st.info("Current training data: synthetic 50-patient dataset for pipeline testing only. Results are not clinical research findings.")
api = st.sidebar.text_input("Backend URL", os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
metadata = {}
health = {}
try:
    health_response = requests.get(f"{api}/health", timeout=2)
    if health_response.ok:
        health = health_response.json()
    metadata_response = requests.get(f"{api}/metadata", timeout=2)
    if metadata_response.ok:
        metadata = metadata_response.json()
    else:
        try:
            detail = metadata_response.json().get("detail", "")
        except (ValueError, requests.RequestException):
            detail = ""
        st.warning(detail or "No trained model is available. Run a training command first; see README.")
except requests.RequestException:
    st.warning("Backend unavailable. Start FastAPI at http://127.0.0.1:8000. Upload preview remains available.")

clinical_available = bool(health.get("clinical_model_loaded", metadata.get("features")))
multimodal_available = bool(health.get("multimodal_model_loaded", metadata.get("multimodal_model_loaded")))
st.subheader("Model Status")
st.table({"Model": ["Clinical model", "ECG encoder", "Image encoder", "Multimodal model"],
          "Status": ["Available" if clinical_available else "Not available",
                     "Available" if multimodal_available else "Not available",
                     "Available" if multimodal_available else "Not available",
                     "Available" if multimodal_available else "Not available"]})

if metadata.get("features") and clinical_available:
    with st.expander("Saved internal test metrics"):
        st.json(metadata.get("metrics", {}).get("test", {}))

values = {}
if metadata.get("features") and clinical_available:
    st.subheader("Clinical inputs")
    with st.form("predict"):
        numeric_features = set(metadata.get("numeric_features", []))
        for feature in metadata["features"]:
            values[feature] = (st.number_input(feature, value=0.0, key=f"field_{feature}")
                               if feature in numeric_features else st.text_input(feature, key=f"field_{feature}"))
        submitted = st.form_submit_button("Predict", type="primary")
else:
    st.info("Prediction fields appear after a trained model is available. The ECG and image preview controls remain usable.")
    submitted = False

if submitted:
    try:
        result = requests.post(f"{api}/predict", json={"clinical": values}, timeout=30)
        if result.ok:
            result = result.json()
            st.subheader("Prediction")
            st.write(f"**{result['prediction']}** · probability {result['probability']:.1%} · confidence {result['confidence']}")
            st.write(f"Decision-boundary uncertainty estimate: {result['uncertainty']:.3f}")
            st.caption("This is a distance-to-boundary score, not a calibrated medical uncertainty estimate.")
            st.subheader("SHAP explanation")
            explanation = result["explanation"]
            if explanation["top_features"]:
                st.dataframe(explanation["top_features"], use_container_width=True)
                chart = pd.DataFrame(explanation["top_features"]).set_index("feature")
                st.bar_chart(chart["shap_value"])
            else:
                st.info(explanation.get("message", "Explanation unavailable."))
        else:
            st.error(result.json().get("detail", "Prediction failed."))
    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")

st.subheader("Optional ECG file")
st.info("Upload a numeric CSV or NPY signal. ECG prediction requires the trained multimodal checkpoint and the other required inputs.")
file_key = f"ecg_upload_{st.session_state.get('ecg_generation', 0)}"
upload = st.file_uploader("Choose ECG file (CSV or NPY)", type=["csv", "npy"], key=file_key)
ecg_valid = False
ecg_signal = None
if upload:
    try:
        ecg_signal, summary = read_ecg_upload(upload.name, upload.getvalue())
        ecg_valid = True
        st.write(f"Selected: {upload.name} ({upload.size} bytes) · {summary}")
        st.line_chart(pd.DataFrame({"ECG": ecg_signal}))
    except Exception as exc:
        st.error(f"Invalid ECG file. Please upload a valid numeric CSV or NPY ECG file. Details: {exc}")
    remove_col, replace_col = st.columns(2)
    with remove_col:
        remove_ecg = st.button("Remove File")
    with replace_col:
        replace_ecg = st.button("Replace File")
    if remove_ecg or replace_ecg:
        st.session_state["ecg_generation"] = st.session_state.get("ecg_generation", 0) + 1
        st.rerun()

st.subheader("Optional medical image")
st.info("Upload a PNG, JPG, or JPEG. Image prediction requires the trained multimodal checkpoint and the other required inputs.")
image_key = f"medical_image_{st.session_state.get('image_generation', 0)}"
medical_image = st.file_uploader("Choose Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key=image_key)
image_valid = False
if medical_image:
    try:
        image_bytes = medical_image.getvalue()
        validate_image_preview(medical_image.name, image_bytes)
        image_valid = True
        st.write(f"Selected: {medical_image.name} ({medical_image.size} bytes)")
        st.image(image_bytes, caption=medical_image.name, use_container_width=True)
    except Exception as exc:
        st.error(f"Invalid medical image: {exc}")
    remove_col, replace_col = st.columns(2)
    with remove_col:
        remove_image = st.button("Remove Image")
    with replace_col:
        replace_image = st.button("Replace Image")
    if remove_image or replace_image:
        st.session_state["image_generation"] = st.session_state.get("image_generation", 0) + 1
        st.rerun()

if multimodal_available:
    st.subheader("Multimodal prediction")
    st.caption("This model was trained on complete clinical, ECG, and image triplets. Provide all three inputs for this workflow.")
    if st.button("Predict CVD", type="primary"):
        if not values or not upload or not ecg_valid or not medical_image or not image_valid:
            st.error("Clinical data, ECG, and medical image are required for multimodal prediction.")
        else:
            try:
                payload = multimodal_payload(values, upload, medical_image)
                response = requests.post(f"{api}/predict/multimodal", json=payload, timeout=60)
                if response.ok:
                    result = response.json()
                    st.write(f"**{result['prediction']}** · probability {result['probability']:.1%}")
                    st.write(f"MC Dropout predictive variance: {result['uncertainty']:.5f} ({result['uncertainty_level']})")
                    st.caption(result["note"])
                else:
                    st.error(response.json().get("detail", "Multimodal prediction failed."))
            except Exception as exc:
                st.error(f"Could not generate multimodal prediction: {exc}")
    if (ecg_valid or image_valid) and values:
        if st.button("Generate Grad-CAM"):
            try:
                payload = multimodal_payload(values, upload if ecg_valid else None,
                                             medical_image if image_valid else None)
                response = requests.post(f"{api}/explain/gradcam", json=payload, timeout=60)
                if not response.ok:
                    st.error(response.json().get("detail", "Grad-CAM unavailable."))
                else:
                    explanation = response.json()
                    heatmap = np.asarray(explanation["heatmap"])[0]
                    if explanation["modality"] == "image":
                        import matplotlib
                        matplotlib.use("Agg", force=True)
                        import matplotlib.pyplot as plt
                        with Image.open(BytesIO(medical_image.getvalue())) as opened:
                            source = np.asarray(opened.convert("RGB").resize((128, 128)))
                        fig, ax = plt.subplots()
                        ax.imshow(source)
                        ax.imshow(heatmap, cmap="jet", alpha=0.4)
                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.line_chart(pd.DataFrame({"ECG": np.asarray(payload["ecg"]),
                                                    "Grad-CAM": np.resize(heatmap, len(payload["ecg"]))}))
            except Exception as exc:
                st.error(f"Grad-CAM unavailable: {exc}")
