# Synthetic CVD Multimodal Dataset — 50 Patients

**SYNTHETIC DATASET — FOR PIPELINE TESTING ONLY.**

This dataset contains 50 synthetic patients with linked clinical data, ECG-like
signals, synthetic PNG images, and synthetic CVD labels. It is not derived from
real patients and must not be used to claim clinical performance.

## Contents
- clinical/clinical.csv — 50 clinical rows
- ecg/ — 50 NPY ECG-like signals (250 Hz, 5 seconds, 1250 samples each)
- images/ — 50 synthetic 224x224 PNG images
- metadata/labels.csv
- metadata/ecg_index.csv
- metadata/image_index.csv
- patient_manifest.csv

All modalities use the same patient_id.

The images are abstract synthetic placeholders, NOT real medical images.
The ECGs are synthetic waveform-like signals, NOT clinical ECG recordings.
The labels are synthetic and do not represent actual disease status.

Use this dataset only to test preprocessing, alignment, training code,
multimodal fusion, SHAP/Grad-CAM/MC-Dropout paths, robustness, subgroups,
FastAPI, Streamlit, and end-to-end integration.
