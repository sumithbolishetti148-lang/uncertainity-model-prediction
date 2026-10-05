import argparse
import json
from pathlib import Path

from ml.multimodal_evaluation import evaluate_logits, load_paired_evaluation

parser = argparse.ArgumentParser(description="External evaluation for a frozen multimodal checkpoint")
parser.add_argument("--clinical", required=True)
parser.add_argument("--ecg-manifest", required=True)
parser.add_argument("--image-manifest", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "multimodal"))
args = parser.parse_args()
model, config, alignment, clinical, ecg, image, y = load_paired_evaluation(
    args.clinical, args.ecg_manifest, args.image_manifest, args.target, args.model_dir,
    require_external_independence=True)
print(json.dumps({"external": True, "training_used": False, "alignment": alignment,
                  "metrics": evaluate_logits(model, clinical, ecg, image, y)}, indent=2))
