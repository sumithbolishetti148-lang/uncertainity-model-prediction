import pandas as pd
import pytest

from ml.pipeline import load_dataset, train, predict_one, evaluate_external
from ml.subgroups import evaluate_subgroups


def test_train_predict_and_missing_feature(tmp_path):
    rows = 120
    frame = pd.DataFrame({
        "age": [30 + i % 50 for i in range(rows)],
        "pressure": [100 + i % 40 for i in range(rows)],
        "smoker": ["yes" if i % 3 else "no" for i in range(rows)],
        "population": ["group_a" if i % 4 < 2 else "group_b" for i in range(rows)],
        "outcome": ["positive" if i % 2 else "negative" for i in range(rows)],
    })
    frame.loc[0, "age"] = None
    data = tmp_path / "sample.csv"
    frame.to_csv(data, index=False)
    model_path = tmp_path / "model.joblib"
    artifact = train(data, "outcome", model_path)
    prediction = predict_one({"age": 55, "pressure": 120, "smoker": "yes", "population": "group_a"}, artifact)
    assert 0 <= prediction["probability"] <= 1
    assert prediction["uncertainty_method"] == "distance_to_decision_boundary"
    with pytest.raises(ValueError, match="Missing required"):
        predict_one({"age": 55, "pressure": 120, "smoker": "yes"}, artifact)
    external = evaluate_external(data, "outcome", artifact)
    assert "pr_auc" in external and "specificity" in external
    subgroup = evaluate_subgroups(data, "outcome", ["population"], model_path, min_group_size=10)
    assert set(subgroup["population"]) == {"group_a", "group_b"}


def test_target_validation(tmp_path):
    data = tmp_path / "invalid.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 0]}).to_csv(data, index=False)
    with pytest.raises(ValueError, match="exactly two"):
        load_dataset(data, "y")
