"""
eda.py
------
Exploratory Data Analysis (EDA) for the Salary Prediction project.

Generates and saves the following to notebooks/eda_outputs/:
    - Dataset summary (printed + saved as text)
    - Missing value analysis (printed + heatmap)
    - Correlation heatmap
    - Salary distribution (histogram + KDE)
    - Box plots (Salary by categorical features)
    - Histograms (all numerical features)
    - Pair plot
    - Feature importance visualization (uses a quick RandomForest fit)

Run from the project root:
    python notebooks/eda.py

NOTE: This script is a plain Python script (not a Jupyter notebook) so
it can run headlessly in any environment, per the "Python only" project
requirement. If you prefer an interactive notebook, you can paste these
same cells into a Jupyter Notebook / VS Code Interactive Window inside
the notebooks/ folder.
"""

import os
import sys

# Allow running this script directly (adds project root to path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import get_clean_dataframe, build_preprocessor, split_features_target
from src.utils import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN, DATASET_PATH

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def dataset_summary(df: pd.DataFrame):
    """Print and save a text summary of the dataset (shape, dtypes, describe)."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nStatistical summary (numerical columns):")
    print(df.describe())
    print("\nStatistical summary (categorical columns):")
    print(df.describe(include="object"))

    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Dtypes:\n" + df.dtypes.to_string() + "\n\n")
        f.write("Numerical summary:\n" + df.describe().to_string() + "\n\n")
        f.write("Categorical summary:\n" + df.describe(include="object").to_string())
    print(f"\nSummary saved to {summary_path}")


def missing_value_analysis(df: pd.DataFrame):
    """Print missing value counts and save a bar chart."""
    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
    print(missing_df if not missing_df.empty else "No missing values found.")

    fig, ax = plt.subplots(figsize=(8, 4))
    df.isnull().sum().plot(kind="bar", ax=ax, color="salmon")
    ax.set_title("Missing Values per Column (Raw Dataset)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "missing_values.png"), bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame):
    """Plot and save a correlation heatmap for numerical features + target."""
    numeric_df = df[NUMERICAL_FEATURES + [TARGET_COLUMN]]
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nCorrelation heatmap saved to {OUTPUT_DIR}/correlation_heatmap.png")


def salary_distribution(df: pd.DataFrame):
    """Plot the distribution of the target variable (Salary)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=40, color="steelblue", ax=ax)
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "salary_distribution.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Salary distribution plot saved to {OUTPUT_DIR}/salary_distribution.png")


def box_plots(df: pd.DataFrame):
    """Box plots of Salary grouped by each categorical feature."""
    for col in CATEGORICAL_FEATURES:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = df.groupby(col)[TARGET_COLUMN].median().sort_values(ascending=False).index
        sns.boxplot(
            data=df, x=col, y=TARGET_COLUMN, order=order,
            hue=col, palette="Set2", legend=False, ax=ax,
        )
        ax.set_title(f"Salary Distribution by {col}")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fname = f"boxplot_salary_by_{col.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight")
        plt.close(fig)
    print(f"Box plots saved to {OUTPUT_DIR}/boxplot_salary_by_*.png")


def histograms(df: pd.DataFrame):
    """Histograms for all numerical features."""
    fig, axes = plt.subplots(1, len(NUMERICAL_FEATURES), figsize=(6 * len(NUMERICAL_FEATURES), 4))
    if len(NUMERICAL_FEATURES) == 1:
        axes = [axes]
    for ax, col in zip(axes, NUMERICAL_FEATURES):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="teal")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "histograms_numerical.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Histograms saved to {OUTPUT_DIR}/histograms_numerical.png")


def pair_plot(df: pd.DataFrame):
    """Pair plot of numerical features + target."""
    subset = df[NUMERICAL_FEATURES + [TARGET_COLUMN]]
    grid = sns.pairplot(subset, diag_kind="kde", plot_kws={"alpha": 0.4})
    grid.fig.suptitle("Pair Plot: Numerical Features vs Salary", y=1.02)
    grid.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"), bbox_inches="tight")
    plt.close(grid.fig)
    print(f"Pair plot saved to {OUTPUT_DIR}/pairplot.png")


def quick_feature_importance(df: pd.DataFrame):
    """
    Fit a quick RandomForestRegressor purely for EDA-stage feature
    importance visualization (separate from the final tuned model
    trained in src/train.py).
    """
    X, y = split_features_target(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_transformed, y)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance_df.head(15), x="Importance", y="Feature",
        hue="Feature", palette="mako", legend=False, ax=ax,
    )
    ax.set_title("EDA-Stage Feature Importance (Quick Random Forest)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_feature_importance.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to {OUTPUT_DIR}/eda_feature_importance.png")


def main():
    print(f"Loading and cleaning dataset from {DATASET_PATH} ...")
    df = get_clean_dataframe()

    dataset_summary(df)
    missing_value_analysis(df)
    correlation_heatmap(df)
    salary_distribution(df)
    box_plots(df)
    histograms(df)
    pair_plot(df)
    quick_feature_importance(df)

    print("\nAll EDA charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
