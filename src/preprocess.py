"""
preprocess.py
-------------
Handles all data preprocessing for the Salary Prediction project:
    1. Loading the raw CSV dataset
    2. Handling missing values
    3. Removing duplicate rows
    4. Detecting and removing outliers (IQR method on the target)
    5. Encoding categorical columns (OneHotEncoder)
    6. Scaling numerical columns (StandardScaler)
    7. Splitting into train/test sets

The fitted encoder + scaler are bundled into a single
ColumnTransformer ("preprocessor") which is saved with joblib so the
exact same transformation can be reapplied at prediction time
(this avoids train/test skew and is a scikit-learn best practice).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.utils import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    get_logger,
)

logger = get_logger(__name__)


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the salary dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(path)
        logger.info("Dataset loaded successfully from %s | shape=%s", path, df.shape)
        return df
    except FileNotFoundError as exc:
        logger.error("Dataset not found at %s", path)
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Run `python dataset/generate_dataset.py` first, or place your "
            "own salary.csv there."
        ) from exc


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.

    Strategy:
        - Numerical columns  -> fill with column median (robust to outliers)
        - Categorical columns -> fill with column mode (most frequent value)
        - Rows missing the target (Salary) are dropped, since we cannot
          train/evaluate on an unknown target.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing target (%s)", dropped, TARGET_COLUMN)

    missing_before = df.isnull().sum().sum()

    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    missing_after = df.isnull().sum().sum()
    logger.info(
        "Missing values handled: %d -> %d remaining", missing_before, missing_after
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataframe."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (%d -> %d)", removed, before, len(df))
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str = TARGET_COLUMN,
                         factor: float = 1.5) -> pd.DataFrame:
    """
    Detect and remove outliers using the Interquartile Range (IQR) method.

    Any row whose `column` value falls outside
        [Q1 - factor * IQR, Q3 + factor * IQR]
    is considered an outlier and removed.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to check for outliers (typically the target, Salary).
    factor : float
        IQR multiplier (1.5 is the standard "mild outlier" threshold).

    Returns
    -------
    pd.DataFrame
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    before = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed = before - len(df_clean)
    logger.info(
        "Outlier removal on '%s': bounds=[%.2f, %.2f] | removed %d rows (%d -> %d)",
        column, lower_bound, upper_bound, removed, before, len(df_clean),
    )
    return df_clean.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
        - Scales numerical features with StandardScaler
        - One-hot encodes categorical features

    Wrapping both in a ColumnTransformer means the SAME fitted object
    can be reused (via joblib) at prediction time, guaranteeing that
    new/unseen input data is transformed identically to the training data.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_clean_dataframe(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Run the full cleaning pipeline (load -> missing values -> duplicates ->
    outliers) and return a clean dataframe ready for feature/target split.
    """
    df = load_dataset(path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, column=TARGET_COLUMN)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Split a clean dataframe into features (X) and target (y).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2, random_state: int = 42):
    """
    Perform a reproducible train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train/test split complete: train=%d rows, test=%d rows",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


def run_full_preprocessing(path: str = DATASET_PATH, test_size: float = 0.2,
                            random_state: int = 42):
    """
    Convenience function that runs the entire preprocessing pipeline
    end-to-end and returns everything train.py needs.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor (unfitted ColumnTransformer)
    """
    df = get_clean_dataframe(path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Quick manual test: run `python -m src.preprocess` from project root
    X_train, X_test, y_train, y_test, preprocessor = run_full_preprocessing()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print(X_train.head())
