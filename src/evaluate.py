"""
evaluate.py
-----------
Evaluation utilities for the Salary Prediction project:
    - Compute standard regression metrics (R2, MAE, MSE, RMSE)
    - Build a model comparison bar chart
    - Build a feature importance chart for tree-based models

Run directly to regenerate charts for an already-trained model:
    python -m src.evaluate
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils import MODEL_DIR, MODEL_PATH, get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute standard regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Label used in the returned dictionary / printed output.

    Returns
    -------
    dict with keys: Model, R2 Score, MAE, MSE, RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "Model": model_name,
        "R2 Score": round(r2, 4),
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
    }

    logger.info(
        "[%s] R2=%.4f | MAE=%.2f | MSE=%.2f | RMSE=%.2f",
        model_name, r2, mae, mse, rmse,
    )
    return metrics


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Create a bar chart comparing R2 scores of all trained models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain 'Model' and 'R2 Score' columns.
    save_path : str, optional
        If given, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(
        data=results_df, x="Model", y="R2 Score", hue="Model",
        palette="viridis", legend=False, ax=axes[0],
    )
    axes[0].set_title("Model Comparison - R2 Score (higher is better)")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=results_df, x="Model", y="RMSE", hue="Model",
        palette="rocket", legend=False, ax=axes[1],
    )
    axes[1].set_title("Model Comparison - RMSE (lower is better)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison chart saved to %s", save_path)

    return fig


def get_feature_importance(pipeline) -> pd.DataFrame:
    """
    Extract feature importances from a fitted Pipeline(preprocessor, regressor).

    Works for tree-based ensembles (Random Forest, Gradient Boosting,
    Extra Trees, XGBoost) which expose `.feature_importances_`.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Must contain steps named "preprocessor" and "regressor".

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance'], sorted descending.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    if not hasattr(regressor, "feature_importances_"):
        raise AttributeError(
            f"{type(regressor).__name__} does not expose feature_importances_"
        )

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                             save_path: str = None):
    """
    Plot a horizontal bar chart of the top N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of get_feature_importance().
    top_n : int
        Number of top features to display.
    save_path : str, optional
        If given, saves the figure to this path.
    """
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    sns.barplot(
        data=top_features, x="Importance", y="Feature", hue="Feature",
        palette="mako", legend=False, ax=ax,
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance chart saved to %s", save_path)

    return fig


if __name__ == "__main__":
    # Regenerate charts from the already-trained model saved on disk.
    from src.preprocess import run_full_preprocessing

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.train` first."
        )

    pipeline = joblib.load(MODEL_PATH)
    _, X_test, _, y_test, _ = run_full_preprocessing()

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name="Saved Model")
    print(metrics)

    importance_df = get_feature_importance(pipeline)
    print(importance_df)

    plot_feature_importance(
        importance_df,
        save_path=os.path.join(MODEL_DIR, "feature_importance.png"),
    )
