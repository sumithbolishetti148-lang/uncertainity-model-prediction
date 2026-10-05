from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import numpy as np
import pandas as pd


def read_ecg_upload(name: str, contents: bytes) -> tuple[np.ndarray, str]:
    """Validate and convert a supported ECG upload to a finite single 1D signal."""
    if len(contents) > 10 * 1024 * 1024:
        raise ValueError("File exceeds the 10 MB limit.")
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(BytesIO(contents), header=None)
        if raw.empty:
            raise ValueError("CSV has no ECG samples.")
        values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    elif suffix == ".npy":
        try:
            values = np.load(BytesIO(contents), allow_pickle=False)
        except Exception as exc:
            raise ValueError("NPY file is corrupted or cannot be read.") from exc
        if values.ndim not in (1, 2) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("NPY must contain a numeric one- or two-dimensional signal.")
        values = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError("Unsupported file type. Choose CSV or NPY.")
    if values.ndim not in (1, 2) or not values.size:
        raise ValueError("ECG must contain numeric samples in one or two dimensions.")
    if not np.isfinite(values).any():
        raise ValueError("CSV has no numeric ECG samples." if suffix == ".csv" else "NPY has no finite numeric ECG samples.")
    if not np.isfinite(values).all():
        raise ValueError("ECG must contain finite numeric samples in one or two dimensions.")
    if values.ndim == 2:
        if min(values.shape) == 0:
            raise ValueError("ECG array has an empty dimension.")
        # Average the lead/channel axis while preserving the longer time axis.
        lead_axis = 0 if values.shape[0] < values.shape[1] else 1
        signal = values.mean(axis=lead_axis)
        descriptor = f"{values.shape[0]}×{values.shape[1]} samples; averaged leads for preview/inference"
    else:
        signal = values
        descriptor = f"{len(signal)} samples"
    if signal.size < 16:
        raise ValueError("ECG must contain at least 16 samples.")
    return np.asarray(signal, dtype=np.float32).reshape(-1), descriptor


def validate_ecg_preview(name: str, contents: bytes) -> str:
    """Validate a CSV/NPY ECG and return a short human-readable signal summary."""
    _signal, descriptor = read_ecg_upload(name, contents)
    return f"Valid numeric ECG: {descriptor}"


def validate_image_preview(name: str, contents: bytes) -> str:
    if len(contents) > 10 * 1024 * 1024:
        raise ValueError("File exceeds the 10 MB limit.")
    if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Unsupported image type. Choose PNG, JPG, or JPEG.")
    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
        with Image.open(BytesIO(contents)) as image:
            if image.width < 1 or image.height < 1:
                raise ValueError("Image dimensions must be positive.")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image contents are invalid or corrupted.") from exc
    return "Image contents are valid"
