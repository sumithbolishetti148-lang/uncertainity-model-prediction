"""Controlled paired-modality perturbations; leaves the source data untouched."""
import argparse
import json
from pathlib import Path

import torch

from ml.multimodal_evaluation import evaluate_logits, load_paired_evaluation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clinical", required=True)
parser.add_argument("--ecg-manifest", required=True)
parser.add_argument("--image-manifest", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "multimodal"))
args = parser.parse_args()
model, config, alignment, clinical, ecg, image, y = load_paired_evaluation(
    args.clinical, args.ecg_manifest, args.image_manifest, args.target, args.model_dir)
generator = torch.Generator().manual_seed(42)
noise = torch.randn(ecg.shape, generator=generator) * 0.05
reports = {"baseline": evaluate_logits(model, clinical, ecg, image, y),
           "ecg_gaussian_noise_sd_0.05": evaluate_logits(model, clinical, ecg + noise, image, y),
           "image_brightness_plus_5pct": evaluate_logits(model, clinical, ecg, (image + 0.05).clamp(0, 1), y),
           "ecg_unavailable": evaluate_logits(model, clinical, None, image, y),
           "image_unavailable": evaluate_logits(model, clinical, ecg, None, y)}
print(json.dumps({"alignment": alignment, "experiments": reports}, indent=2))
