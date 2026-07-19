"""
test_preprocess.py
-------------------
Unit tests for src/preprocess.py: missing value handling, duplicate
removal, outlier removal, and the preprocessing ColumnTransformer.
"""

import pandas as pd

from src.preprocess import (
    build_preprocessor,
    handle_missing_values,
    remove_duplicates,
    remove_outliers_iqr,
    split_features_target,
    train_test_split_data,
)
from src.utils import ALL_FEATURES, TARGET_COLUMN


def test_handle_missing_values_fills_all_gaps(sample_raw_df):
    cleaned = handle_missing_values(sample_raw_df)
    # No missing values should remain in any feature column
    assert cleaned[ALL_FEATURES].isnull().sum().sum() == 0


def test_handle_missing_values_drops_rows_missing_target():
    df = pd.DataFrame({
        "Age": [25, 30],
        "Gender": ["Male", "Female"],
        "Education Level": ["Bachelor's", "Master's"],
        "Job Title": ["Data Analyst", "Data Scientist"],
        "Years of Experience": [2.0, 5.0],
        "City": ["Austin", "Seattle"],
        "Salary": [55000.0, None],
    })
    cleaned = handle_missing_values(df)
    assert len(cleaned) == 1
    assert cleaned[TARGET_COLUMN].isnull().sum() == 0


def test_remove_duplicates_removes_exact_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    result = remove_duplicates(df)
    assert len(result) == 2


def test_remove_outliers_iqr_removes_extreme_salary(sample_raw_df):
    cleaned = handle_missing_values(sample_raw_df)
    result = remove_outliers_iqr(cleaned, column=TARGET_COLUMN)
    # The 5,000,000 salary outlier should be gone
    assert result[TARGET_COLUMN].max() < 1_000_000
    assert len(result) < len(cleaned)


def test_build_preprocessor_transforms_expected_shape(sample_raw_df):
    cleaned = handle_missing_values(sample_raw_df)
    X, y = split_features_target(cleaned)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Rows preserved, columns expanded (numeric + one-hot categorical)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > len(ALL_FEATURES)


def test_train_test_split_data_respects_test_size(sample_raw_df):
    cleaned = handle_missing_values(sample_raw_df)
    X, y = split_features_target(cleaned)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.5, random_state=42
    )
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
