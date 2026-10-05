# Explainable and Uncertainty-Aware Multimodal Deep Learning for Generalizable Cardiovascular Disease Prediction Across Heterogeneous Patient Populations

This repository combines a Paper 1 multimodal research pipeline with Paper 2 evaluation and interpretability interfaces. The workspace includes a 50-patient **synthetic dataset for software pipeline testing only** and a checkpoint trained from it. Neither the data nor its metrics represent patients or clinical research findings. No real clinical dataset or clinically validated model is available.

> This system is intended for academic and research purposes only and is not a medical diagnostic tool. Predictions should not be used for clinical decision-making.

## Research architecture

```text
Clinical CSV ── train-only imputation/encoding/scaling ── clinical MLP ─┐
ECG CSV/NPY ── resample/normalize ── 1D CNN ────────────────────────────┼─
PNG/JPEG ── RGB resize/normalize ── 2D CNN ────────────────────────────┘
             ID-aligned embeddings → symmetric cross-modal InfoNCE
             → availability-aware concatenation fusion → CVD head

Paper 2: SHAP (clinical baseline) · MC Dropout · prediction-targeted Grad-CAM
         independent external evaluation · controlled robustness · subgroup metrics
```

Contrastive positives are pairs from the same unique patient ID; other distinct patients in a batch are negatives. The loss is symmetric pairwise InfoNCE averaged over all available modality pairs. It is combined with classification cross-entropy as `CE + contrastive_weight × InfoNCE`. The fusion head concatenates three embeddings and three modality-presence indicators. Encoders are intentionally small; the default embedding size is 64.

## Current state

- The original clinical CSV pipeline, balanced Logistic Regression baseline, FastAPI, Streamlit UI, local SHAP, external clinical evaluation, robustness command, and tests are preserved.
- Added ID-keyed clinical/ECG/image alignment, CSV/NPY ECG reader, PNG/JPEG reader, trainable clinical/ECG/image encoders, unimodal neural comparisons, multimodal fusion, contrastive training, model checkpoints, MC Dropout, ECG/image Grad-CAM, paired external evaluation, multimodal robustness, subgroup analysis, and image upload/removal controls.
- When the paired multimodal checkpoint exists, the UI accepts all three modalities for multimodal prediction, shows ECG/image previews, provides MC Dropout predictive variance and supports image/ECG Grad-CAM. A trained clinical baseline also supports clinical-only prediction with SHAP.
- Image formats supported now: PNG, JPG, JPEG. ECG formats: numeric CSV and NPY. DICOM and WFDB are not implemented.
- The confidence and uncertainty scores are model estimates for research, not medical certainty or calibrated clinical risk.

## Dataset and patient alignment

The multimodal training command accepts:

1. A clinical CSV with one row per patient, a configured patient ID (default `patient_id`), a binary target, and actual clinical features.
2. An ECG manifest CSV with `patient_id,ecg_path` (or `patient_id,ecg_file`).
3. An image manifest CSV with `patient_id,image_path` (or `patient_id,image_file`).

Manifest paths are relative to the manifest file's directory. Each modality manifest currently allows at most one file per patient; duplicate IDs are rejected with a clear error. Alignment is by patient ID, never row order. The alignment report lists missing and unmatched patients and complete triplets. Multimodal training uses only complete triplets and performs fixed-seed patient-level 70/15/15 stratified splits before fitting clinical preprocessing. There must be at least seven complete records per target class and each split must contain both classes.

The included synthetic files live under `ml/data/`: `clinical/clinical.csv`, 50 one-dimensional 1,250-sample NPY signals in `ecg/`, 50 224×224 PNG images in `images/`, and `patient_manifest.csv`. The shared manifest uses `ecg_file` and `image_file`; both can be passed directly to the trainer as their respective manifests. For pipeline tests, run:

```bash
python3 -m ml.train_multimodal \
  --clinical ml/data/clinical/clinical.csv --target cvd_label \
  --ecg-manifest ml/data/patient_manifest.csv \
  --image-manifest ml/data/patient_manifest.csv \
  --output models/multimodal --epochs 3 --batch-size 8 \
  --embedding-dim 16 --ecg-samples 1250 --image-size 128
```

This command writes the multimodal checkpoint and clinical baseline to `models/multimodal/`. Its metrics are software fixtures only and must not be cited as clinical performance.

ECG CSV/NPY must contain numeric signal samples; 2D multi-lead arrays are averaged to one signal, interpolated to 5,000 samples by default, and z-normalized. Images are decoded, converted to RGB, and resized to 128×128. No signal is converted into an image. Unsupported formats fail validation. Dataset files and model artifacts are ignored by git.

Example layout:

```text
data/
  clinical.csv
  ecg_manifest.csv
  ecg/patient_a.csv
  image_manifest.csv
  images/patient_a.png
models/                    generated model artifacts
```

## Install

Python 3.12 is tested in this workspace; Python 3.11–3.12 are supported by the pinned dependency set. PyTorch is included for the multimodal model. No Docker or npm is required.

macOS/Linux, from the repository root:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip check
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If `python` is not found, use `python3` on macOS/Linux or `py -3.12` on Windows. If `cd` says the directory is missing, check `pwd` / `Get-Location` and use the full repository path in quotes.

## Train the clinical baseline

```bash
python3 -m ml.pipeline --data data/clinical.csv --target cvd_label
```

Replace `cvd_label` with the real target column. The loader discovers actual feature columns, handles missing values and numeric infinities, fits preprocessing on the training partition only, and saves `models/clinical_model.joblib` plus its metrics JSON. A positive class is inferred for conventional labels (`0/1`, `yes/no`, `positive/negative`, `case/control`, CVD/no-CVD); otherwise supply `--positive-label VALUE`.

## Train the multimodal Paper 1 model

```bash
python3 -m ml.train_multimodal \
  --clinical data/clinical.csv \
  --target cvd_label \
  --ecg-manifest data/ecg_manifest.csv \
  --image-manifest data/image_manifest.csv \
  --output models/multimodal
```

Optional controls include `--patient-id`, `--positive-label`, `--epochs`, `--batch-size`, `--embedding-dim`, `--ecg-samples`, and `--image-size`. It trains and compares clinical-only, ECG-only, image-only, the multimodal model, and the preserved clinical Logistic Regression baseline on the same patient split. It saves weights, preprocessing, configuration, alignment report, and internal validation/test metrics under `models/multimodal/`. No training runs at API startup.

## Evaluate and explain

Clinical independent external validation:

```bash
python3 -m ml.evaluate_external --data data/external.csv --target cvd_label
```

Multimodal independent external evaluation (requires separate manifests and must not reuse training patients):

```bash
python3 -m ml.evaluate_multimodal_external \
  --clinical data/external/clinical.csv --target cvd_label \
  --ecg-manifest data/external/ecg_manifest.csv \
  --image-manifest data/external/image_manifest.csv \
  --model-dir models/multimodal
```

Clinical baseline robustness and subgroup reports:

```bash
python3 -m ml.robustness --data data/clinical.csv --target cvd_label
python3 -m ml.subgroups --data data/clinical.csv --target cvd_label --group-columns sex age_group site
```

Multimodal subgroup analysis uses only the patient-level internal test IDs saved with the trained checkpoint. Supply only group columns that exist in the clinical CSV:

```bash
python3 -m ml.subgroups_multimodal \
  --clinical data/clinical.csv --target cvd_label \
  --ecg-manifest data/ecg_manifest.csv \
  --image-manifest data/image_manifest.csv \
  --group-columns sex age_group site --model-dir models/multimodal
```

Groups without the configured minimum sample size or both target classes are reported as insufficient rather than assigned performance metrics. This evaluates the same Paper 1 multimodal checkpoint used for Paper 2 external and robustness analyses.

Multimodal ECG-noise, image-brightness, and missing-modality checks:

```bash
python3 -m ml.robustness_multimodal \
  --clinical data/clinical.csv --target cvd_label \
  --ecg-manifest data/ecg_manifest.csv \
  --image-manifest data/image_manifest.csv --model-dir models/multimodal
```

All evaluation commands require the corresponding real data and trained artifacts. The subgroup command reports insufficient samples rather than fabricating subgroup results. External test rows never enter the training path.

The paired multimodal trainer stores the IDs used in its training dataset. The multimodal external-evaluation command checks those IDs against every patient in the submitted clinical table and rejects any overlap. The clinical CSV external evaluator and API apply the same check when the training artifact has patient IDs; external files must include those IDs. A clinical model trained without patient IDs cannot prove patient-level independence, so its report explicitly marks overlap as not verifiable. Independent source/site provenance still needs to be established by the researcher.

For local per-patient SHAP, use the clinical prediction API/UI. Generate a global SHAP summary plot for the saved clinical Logistic Regression model with:

```bash
python3 -m ml.explain_global --output models/shap_summary.png
```

Multimodal Grad-CAM is computed from the predicted class through the corresponding CNN and fusion output. The backend returns a heatmap; the UI overlays image heatmaps on the uploaded image or plots an ECG heatmap with the signal. Grad-CAM is unavailable for a modality that was not provided or for a missing multimodal model.

MC Dropout runs repeated stochastic inference with dropout active, returning mean probability and predictive variance. It is a model uncertainty estimate, not a clinical certainty score.

## Backend and frontend (separate ports)

Create `.env` from `.env.example` or set `BACKEND_URL` in the shell. In terminal 1, from the repository root:

```bash
source .venv/bin/activate
python3 -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In terminal 2, also from the repository root:

```bash
source .venv/bin/activate
BACKEND_URL=http://127.0.0.1:8000 python3 -m streamlit run frontend/app.py --server.port 8501
```

The Streamlit frontend listens on **127.0.0.1:8501** and calls FastAPI on **127.0.0.1:8000**. Windows PowerShell activation uses `.venv\Scripts\Activate.ps1`; set the URL with `$env:BACKEND_URL="http://127.0.0.1:8000"` and use `python -m ...`.

Model paths are anchored to the repository root. `CVD_MODEL_PATH=models/clinical_model.joblib` and `CVD_MULTIMODAL_MODEL_DIR=models/multimodal` in `.env` therefore resolve consistently even if the backend is launched from another working directory. Multimodal training writes to the same `models/multimodal` directory by default. If no checkpoint exists, `/health` reports the model as unavailable, `/metadata` returns an actionable training message, and Streamlit remains open with upload controls instead of stopping.

Implemented API endpoints:

- `GET /health` — service and artifact availability.
- `GET /metadata` — discovered features and saved metrics.
- `POST /predict` — clinical Logistic Regression prediction and local SHAP when that artifact exists.
- `POST /predict/multimodal` — trained multimodal prediction plus MC Dropout variance. JSON fields: `clinical`, optional numeric `ecg` vector, optional normalized RGB `image` array, optional `passes`.
- `POST /explain/gradcam` — prediction-targeted 1D ECG or 2D image Grad-CAM for the trained multimodal model.

Open `http://127.0.0.1:8000/docs` for API schemas. Missing artifacts return 503; invalid features/signals/images return a client error. Set `CVD_MODEL_PATH` or `CVD_MULTIMODAL_MODEL_DIR` to use artifact directories outside defaults.

## Tests and verification

```bash
python3 -m pytest -q
python3 -m pip check
python3 -m compileall .
```

Tests use temporary CSVs, numeric signals, and temporary images to verify software behavior, patient-ID alignment, model creation/checkpoint reload, training, prediction, SHAP, MC Dropout, Grad-CAM, file validation/removal, API, and frontend connection. These fixtures are not real medical records and their metrics are not project results.

## Troubleshooting

- **`python` not found:** use `python3` or Windows `py -3.12`.
- **`npm not found`:** npm is not used; the UI is Streamlit.
- **Dependency conflict/install error:** use the documented Python version in a fresh `.venv`, then reinstall `requirements.txt`.
- **Port conflict:** preserve separate ports; change the backend port and set the same `BACKEND_URL` for Streamlit.
- **Backend unavailable:** check FastAPI at `127.0.0.1:8000` and the frontend's `BACKEND_URL`.
- **Model missing:** run the corresponding training command first.
- **Manifest alignment error:** ensure IDs match exactly and paths resolve relative to their manifest. Duplicate per-patient files are currently rejected.
- **No complete triplets:** confirm the patient keys and modality manifests; the training command intentionally excludes incomplete triplets and prints alignment counts.
- **Unsupported medical image:** use PNG/JPG/JPEG; DICOM is not yet supported.
- **SHAP unavailable:** prediction continues with an explicit explainer message; no attribution is fabricated.

## Current data status and limitations

| Component | Implemented | Tested | Real dataset required | Notes |
|---|---|---|---|---|
| Clinical preprocessing | Yes | Yes | Yes for research use | Train-only transforms; synthetic data executed |
| ECG preprocessing | Yes | Yes | Yes for research use | CSV/NPY numeric signals; synthetic 1D NPY executed |
| Image preprocessing | Yes | Yes | Yes for research use | PNG/JPG/JPEG; synthetic PNG executed |
| Clinical/ECG/image encoders | Yes | Yes | Yes for research | Small MLP, 1D CNN, 2D CNN |
| Cross-modal contrastive learning | Yes | Yes | Yes for meaningful research | ID-aligned symmetric InfoNCE |
| Multimodal fusion/prediction | Yes | Yes, synthetic data | Yes for research metrics | Concatenation plus presence flags |
| Logistic and unimodal baselines | Yes | Yes, synthetic data | Yes for research metrics | Same patient split in multimodal runner |
| SHAP | Yes | Yes, synthetic input | Trained clinical model | Local API and global summary script |
| Grad-CAM | Yes | Yes, synthetic image/ECG | Trained multimodal model | Image overlay and ECG-compatible map |
| MC Dropout uncertainty | Yes | Yes, synthetic input | Trained neural model | Predictive variance; not calibrated clinical risk |
| External validation | Yes | Pipeline tested | Independent dataset | No independent set supplied/results absent |
| Robustness | Yes | Yes, synthetic data | Yes for research | Controlled perturbation checks; synthetic metrics are not research findings |
| Heterogeneous subgroups | Yes | Yes, synthetic data | Yes for research | Small held-out multimodal groups reported insufficient |
| FastAPI / Streamlit | Yes | Yes | No for software use | Local services at ports 8000 / 8501 |
| ECG file remove and inference | Yes | Yes | No for pipeline checks | CSV/NPY validated; full-triplet prediction enabled |
| Image file remove and inference | Yes | Yes | No for pipeline checks | PNG/JPG/JPEG validated; preview and Grad-CAM enabled |

**Synthetic dataset available: YES (50 patients). Synthetic model training and local prediction paths: VERIFIED. Real clinical dataset: NOT AVAILABLE. Clinical, external, robustness, subgroup, and generalization conclusions: NOT ESTABLISHED.**

## Medical disclaimer

> This system is intended for academic and research purposes only and is not a medical diagnostic tool. Predictions should not be used for clinical decision-making.
