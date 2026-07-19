"""
test_predict.py
----------------
Unit tests for src/predict.py: input validation logic.

These tests focus on validate_inputs() because it has no dependency on
a trained model file, so they run reliably in any environment (including
a fresh CI checkout where models/salary_model.pkl doesn't exist yet).
"""

import pytest

from src.predict import InvalidInputError, validate_inputs


def test_validate_inputs_accepts_valid_data():
    age, experience = validate_inputs(
        age=30, gender="Male", education="Bachelor's",
        experience=5, job_title="Data Analyst", city="Austin",
    )
    assert age == 30
    assert experience == 5


def test_validate_inputs_rejects_invalid_age():
    with pytest.raises(InvalidInputError):
        validate_inputs(
            age=150, gender="Male", education="Bachelor's",
            experience=5, job_title="Data Analyst", city="Austin",
        )


def test_validate_inputs_rejects_unknown_category():
    with pytest.raises(InvalidInputError):
        validate_inputs(
            age=30, gender="Unknown", education="Bachelor's",
            experience=5, job_title="Data Analyst", city="Austin",
        )


def test_validate_inputs_rejects_unrealistic_experience_for_age():
    with pytest.raises(InvalidInputError):
        validate_inputs(
            age=20, gender="Male", education="Bachelor's",
            experience=30, job_title="Data Analyst", city="Austin",
        )


def test_validate_inputs_rejects_non_numeric_age():
    with pytest.raises(InvalidInputError):
        validate_inputs(
            age="thirty", gender="Male", education="Bachelor's",
            experience=5, job_title="Data Analyst", city="Austin",
        )
