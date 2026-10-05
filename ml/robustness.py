"""Small controlled missing-value and numeric perturbation checks."""
import argparse
import json

import numpy as np

from ml.pipeline import load_dataset, load_model, _metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--model", default=None)
args = parser.parse_args()
artifact = load_model(args.model)
X, y = load_dataset(args.data, args.target)
X = X[artifact["features"]]
baseline = _metrics(artifact["model"], X, y, artifact["positive_class"])
rng = np.random.default_rng(42)
numeric = artifact["numeric_features"]
missing = X.copy()
if numeric:
    missing.loc[:, numeric[0]] = np.nan
missing_result = _metrics(artifact["model"], missing, y, artifact["positive_class"])
perturbed = X.copy()
for col in numeric:
    scale = perturbed[col].std(skipna=True)
    if np.isfinite(scale) and scale > 0:
        perturbed[col] = perturbed[col] + rng.normal(0, 0.05 * scale, len(perturbed))
perturb_result = _metrics(artifact["model"], perturbed, y, artifact["positive_class"])
print(json.dumps({"baseline": baseline, "one_numeric_feature_missing": missing_result,
                  "numeric_perturbation_0.05_sd": perturb_result}, indent=2))
