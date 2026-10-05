"""Evaluate the clinical baseline across explicitly supplied, real subgroup columns."""
import argparse
import json

import pandas as pd

from ml.pipeline import _metrics, load_model


def evaluate_subgroups(dataset, target, group_columns, model_path=None, min_group_size=10):
    artifact = load_model(model_path)
    frame = pd.read_csv(dataset)
    required = set(artifact["features"]) | {target} | set(group_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required model/group columns: {missing}")
    result = {}
    for column in group_columns:
        result[column] = {}
        for value, group in frame.dropna(subset=[column, target]).groupby(column, dropna=True):
            key = str(value)
            if len(group) < min_group_size or group[target].nunique() != 2:
                result[column][key] = {"status": "insufficient_data", "n": int(len(group)),
                                       "minimum": min_group_size,
                                       "reason": "requires minimum rows and both target classes"}
                continue
            result[column][key] = _metrics(artifact["model"], group[artifact["features"]],
                                           group[target], artifact["positive_class"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-columns", nargs="+", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-group-size", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate_subgroups(args.data, args.target, args.group_columns,
                                        args.model, args.min_group_size), indent=2))
