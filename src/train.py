"""
train.py
--------
Trains and compares several ensemble regression models for salary
prediction:
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Extra Trees Regressor
    - XGBoost Regressor (only if the `xgboost` package is installed)

The best-performing model (by R^2 score on the test set) is then
tuned with RandomizedSearchCV and saved to disk with joblib, along
with the fitted preprocessing pipeline.

Run directly:
    python -m src.train
"""

import time

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.evaluate import evaluate_model
from src.preprocess import run_full_preprocessing
from src.utils import (
    METRICS_PATH,
    MODEL_PATH,
    ensure_dirs,
    get_logger,
)

logger = get_logger(__name__)

# Try to import XGBoost; it's an optional dependency per the assignment spec
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost is not installed - skipping XGBoost Regressor. "
        "Install it with `pip install xgboost` to include it in the comparison."
    )


def get_candidate_models() -> dict:
    """
    Build a dictionary of candidate ensemble models to train and compare.

    Returns
    -------
    dict[str, estimator]
    """
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    return models


def train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train every candidate model inside a Pipeline(preprocessor + model),
    evaluate each on the test set, and return a results table plus the
    dictionary of fitted pipelines.

    Returns
    -------
    results_df : pd.DataFrame
        Comparison table sorted by R2 score (descending).
    fitted_pipelines : dict[str, Pipeline]
    """
    models = get_candidate_models()
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        logger.info("Training model: %s", name)
        start = time.time()

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - start
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_name=name)
        metrics["Training Time (s)"] = round(elapsed, 2)

        results.append(metrics)
        fitted_pipelines[name] = pipeline

        logger.info(
            "%s trained in %.2fs | R2=%.4f | RMSE=%.2f",
            name, elapsed, metrics["R2 Score"], metrics["RMSE"],
        )

    results_df = pd.DataFrame(results).sort_values(
        by="R2 Score", ascending=False
    ).reset_index(drop=True)

    return results_df, fitted_pipelines


def tune_best_model(best_model_name: str, X_train, y_train, preprocessor):
    """
    Run RandomizedSearchCV on the best-performing model to squeeze out
    additional performance via hyperparameter tuning.

    Parameters
    ----------
    best_model_name : str
        One of "Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost"
    X_train, y_train : training data
    preprocessor : unfitted ColumnTransformer

    Returns
    -------
    best_pipeline : Pipeline
        The tuned pipeline, refit on the full training set.
    best_params : dict
    """
    param_grids = {
        "Random Forest": {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__max_depth": [None, 5, 10, 15, 20],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [2, 3, 4, 5],
            "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        },
        "Extra Trees": {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__max_depth": [None, 5, 10, 15, 20],
            "regressor__min_samples_split": [2, 5, 10],
        },
    }

    if XGBOOST_AVAILABLE:
        param_grids["XGBoost"] = {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [3, 4, 5, 6],
            "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
            "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

    base_models = get_candidate_models()
    base_model = base_models[best_model_name]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", base_model),
    ])

    param_grid = param_grids.get(best_model_name, {})
    if not param_grid:
        logger.warning(
            "No tuning grid defined for %s; returning untuned pipeline.",
            best_model_name,
        )
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    logger.info("Starting RandomizedSearchCV for %s ...", best_model_name)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=15,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("Best params for %s: %s", best_model_name, search.best_params_)
    logger.info("Best CV R2 score: %.4f", search.best_score_)

    return search.best_estimator_, search.best_params_


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    """Persist a fitted pipeline (preprocessor + model) to disk with joblib."""
    ensure_dirs()
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)


def main():
    """Full training pipeline: preprocess -> train -> compare -> tune -> save."""
    ensure_dirs()

    logger.info("=== Starting Salary Prediction training pipeline ===")
    X_train, X_test, y_train, y_test, preprocessor = run_full_preprocessing()

    logger.info("Training and comparing ensemble models ...")
    results_df, fitted_pipelines = train_and_compare_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70 + "\n")

    results_df.to_csv(METRICS_PATH, index=False)
    logger.info("Comparison table saved to %s", METRICS_PATH)

    best_model_name = results_df.iloc[0]["Model"]
    logger.info("Best baseline model: %s", best_model_name)

    logger.info("Tuning best model with RandomizedSearchCV ...")
    tuned_pipeline, best_params = tune_best_model(
        best_model_name, X_train, y_train, preprocessor
    )

    # Evaluate the tuned model on the held-out test set
    y_pred_tuned = tuned_pipeline.predict(X_test)
    tuned_metrics = evaluate_model(y_test, y_pred_tuned, model_name=f"{best_model_name} (Tuned)")

    print("\n" + "=" * 70)
    print(f"TUNED MODEL PERFORMANCE: {best_model_name}")
    print("=" * 70)
    for k, v in tuned_metrics.items():
        print(f"{k:>15}: {v}")
    print("=" * 70 + "\n")

    save_model(tuned_pipeline)

    logger.info("=== Training pipeline complete ===")
    return results_df, tuned_pipeline, tuned_metrics


if __name__ == "__main__":
    main()
