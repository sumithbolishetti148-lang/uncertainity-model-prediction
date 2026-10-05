"""Patient-ID alignment and readers for paired clinical, ECG, and image data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _manifest(frame: pd.DataFrame | None, modality: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["patient_id", f"{modality}_path"])
    path_col = f"{modality}_path"
    file_col = f"{modality}_file"
    if path_col not in frame and file_col in frame:
        frame = frame.rename(columns={file_col: path_col})
    missing = {"patient_id", path_col} - set(frame.columns)
    if missing:
        raise ValueError(f"{modality} manifest is missing columns: {sorted(missing)}")
    if frame["patient_id"].isna().any() or frame["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{modality} manifest contains missing patient IDs")
    if frame[path_col].isna().any() or frame[path_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"{modality} manifest contains missing file paths")
    duplicates = frame.loc[frame["patient_id"].duplicated(keep=False), "patient_id"].astype(str).unique()
    if len(duplicates):
        raise ValueError(f"Multiple {modality} files per patient are unsupported; duplicate IDs: {duplicates[:8].tolist()}")
    return frame[["patient_id", path_col]].copy()


def align_patient_modalities(
    clinical: pd.DataFrame,
    ecg_manifest: pd.DataFrame | None = None,
    image_manifest: pd.DataFrame | None = None,
    *,
    clinical_id: str = "patient_id",
    ecg_manifest_dir: str | Path = ".",
    image_manifest_dir: str | Path = ".",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-align manifests by explicit patient IDs; never align on row order."""
    if clinical_id not in clinical:
        raise ValueError(f"Clinical table requires the configured patient ID column '{clinical_id}'")
    if clinical[clinical_id].isna().any() or clinical[clinical_id].astype(str).str.strip().eq("").any():
        raise ValueError("Clinical table contains missing patient IDs")
    if clinical[clinical_id].duplicated().any():
        dup = clinical.loc[clinical[clinical_id].duplicated(keep=False), clinical_id].astype(str).unique()
        raise ValueError(f"Clinical table must have one row per patient; duplicates: {dup[:8].tolist()}")
    base = clinical.rename(columns={clinical_id: "patient_id"}).copy()
    ecg = _manifest(ecg_manifest, "ecg")
    images = _manifest(image_manifest, "image")
    base["patient_id"] = base.patient_id.astype(str)
    ecg["patient_id"] = ecg.patient_id.astype(str)
    images["patient_id"] = images.patient_id.astype(str)
    clinical_ids = set(base.patient_id.astype(str))
    ecg_ids = set(ecg.patient_id.astype(str))
    image_ids = set(images.patient_id.astype(str))
    aligned = base.merge(ecg, on="patient_id", how="left", validate="one_to_one")
    aligned = aligned.merge(images, on="patient_id", how="left", validate="one_to_one")

    def resolve_files(frame: pd.DataFrame, column: str, directory: str | Path):
        root = Path(directory)
        for value in frame[column].dropna():
            full = (root / str(value)).resolve()
            if not full.is_file():
                raise FileNotFoundError(f"Manifest file not found: {full}")
        return frame[column].map(lambda p: str((root / str(p)).resolve()) if pd.notna(p) else None)

    aligned["ecg_path"] = resolve_files(aligned, "ecg_path", ecg_manifest_dir)
    aligned["image_path"] = resolve_files(aligned, "image_path", image_manifest_dir)
    aligned["has_ecg"] = aligned.ecg_path.notna()
    aligned["has_image"] = aligned.image_path.notna()
    report = {
        "clinical_patients": len(clinical_ids),
        "ecg_patients": len(ecg_ids),
        "image_patients": len(image_ids),
        "missing_ecg_patient_ids": sorted(clinical_ids - ecg_ids),
        "missing_image_patient_ids": sorted(clinical_ids - image_ids),
        "unmatched_ecg_patient_ids": sorted(ecg_ids - clinical_ids),
        "unmatched_image_patient_ids": sorted(image_ids - clinical_ids),
        "complete_triplets": int((aligned.has_ecg & aligned.has_image).sum()),
        "clinical_only": int((~aligned.has_ecg & ~aligned.has_image).sum()),
        "missing_patient_ids": 0,
    }
    return aligned, report


def load_ecg(path: str | Path, samples: int = 5000) -> np.ndarray:
    """Read numeric CSV or NPY ECG and resample to a fixed-length 1D vector."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        signal = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        signal = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy()
    else:
        raise ValueError(f"Unsupported ECG format '{path.suffix}'. Supported formats: CSV, NPY")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2:
        if signal.shape[1] > signal.shape[0]:
            signal = signal.T
        signal = np.nanmean(signal, axis=0)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).any():
        raise ValueError(f"ECG file must contain a finite numeric signal: {path}")
    signal = np.nan_to_num(signal, nan=float(np.nanmean(signal)))
    # Linear interpolation preserves the signal domain; no image conversion.
    old_x = np.linspace(0, 1, num=signal.size)
    new_x = np.linspace(0, 1, num=samples)
    signal = np.interp(new_x, old_x, signal).astype(np.float32)
    scale = float(signal.std())
    if scale > 1e-8:
        signal = (signal - float(signal.mean())) / scale
    return signal


def load_medical_image(path: str | Path, size: int = 128) -> np.ndarray:
    """Load PNG/JPEG as normalized RGB CHW; DICOM requires separate clinical review."""
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format '{path.suffix}'. Supported formats: PNG, JPG, JPEG")
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))
