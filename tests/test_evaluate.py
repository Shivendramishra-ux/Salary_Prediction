"""
test_evaluate.py
-----------------
Unit tests for src/evaluate.py: regression metric computation.
"""

import numpy as np

from src.evaluate import evaluate_model


def test_evaluate_model_perfect_prediction():
    y_true = [100, 200, 300, 400]
    y_pred = [100, 200, 300, 400]
    metrics = evaluate_model(y_true, y_pred, model_name="Perfect")

    assert metrics["R2 Score"] == 1.0
    assert metrics["MAE"] == 0.0
    assert metrics["MSE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["Model"] == "Perfect"


def test_evaluate_model_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    metrics = evaluate_model(y_true, y_pred, model_name="Offset by 10")

    # Both predictions are off by exactly 10
    assert metrics["MAE"] == 10.0
    assert metrics["MSE"] == 100.0
    assert metrics["RMSE"] == 10.0


def test_evaluate_model_returns_expected_keys():
    metrics = evaluate_model([1, 2, 3], [1, 2, 4], model_name="X")
    expected_keys = {"Model", "R2 Score", "MAE", "MSE", "RMSE"}
    assert expected_keys.issubset(metrics.keys())
