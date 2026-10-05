"""Generate a real global SHAP summary plot from the trained clinical artifact."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import shap

from ml.pipeline import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default=None)
parser.add_argument("--output", default="models/shap_summary.png")
args = parser.parse_args()
artifact = load_model(args.model)
pipeline = artifact["model"]
preprocessor = pipeline.named_steps["preprocess"]
background = preprocessor.transform(artifact["shap_background"])
sample = preprocessor.transform(artifact["shap_background"])
if hasattr(background, "toarray"):
    background = background.toarray()
    sample = sample.toarray()
explainer = shap.LinearExplainer(pipeline.named_steps["model"], background)
values = np.asarray(explainer.shap_values(sample))
if values.ndim == 3:
    values = values[:, :, -1]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
shap.summary_plot(values, sample, feature_names=preprocessor.get_feature_names_out(), show=False)
plt.tight_layout()
plt.savefig(args.output, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved SHAP summary plot: {args.output} (rows={sample.shape[0]})")
