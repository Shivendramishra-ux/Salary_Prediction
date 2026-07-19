"""
conftest.py
-----------
Shared pytest fixtures for the test suite. Ensures the project root is
importable (so `from src... import ...` works regardless of where
pytest is invoked from) and provides a small reusable sample dataframe.
"""

import os
import sys

import pandas as pd
import pytest

# Make sure the project root is on sys.path so `import src` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A tiny, deliberately messy dataframe mirroring the real dataset schema."""
    return pd.DataFrame({
        "Age": [25, 30, None, 45, 25, 200],
        "Gender": ["Male", "Female", "Male", None, "Male", "Female"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Bachelor's", "PhD"],
        "Job Title": [
            "Data Analyst", "Data Scientist", "Software Engineer",
            "Product Manager", "Data Analyst", "Research Scientist",
        ],
        "Years of Experience": [2.0, 5.0, 10.0, 15.0, 2.0, 3.0],
        "City": ["Austin", "Seattle", "Boston", "Remote", "Austin", "Denver"],
        # Last row is an extreme salary outlier on purpose
        "Salary": [55000.0, 90000.0, 130000.0, 140000.0, 55000.0, 5_000_000.0],
    })
