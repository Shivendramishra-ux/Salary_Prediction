"""
predict.py
----------
Loads the trained, saved model (models/salary_model.pkl) and predicts
the expected salary for a new individual based on:
    - Age
    - Gender
    - Education Level
    - Years of Experience
    - Job Title
    - City

Can be used in two ways:

1. As an importable function inside app.py or other scripts:
       from src.predict import predict_salary
       salary = predict_salary(age=30, gender="Male", education="Master's",
                                experience=5, job_title="Data Scientist",
                                city="Austin")

2. As an interactive command-line script:
       python -m src.predict
"""

import os

import joblib
import pandas as pd

from src.utils import ALL_FEATURES, MODEL_PATH, get_logger

logger = get_logger(__name__)

# Valid categorical options — kept in sync with dataset/generate_dataset.py.
# When using a real-world dataset, update these lists (or load them
# dynamically from the training data) to match your data's categories.
VALID_GENDERS = ["Male", "Female"]
VALID_EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
VALID_JOB_TITLES = [
    "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
    "HR Manager", "Sales Executive", "Marketing Manager", "Business Analyst",
    "Financial Analyst", "Operations Manager", "Project Manager",
    "UX Designer", "DevOps Engineer", "Customer Support Specialist",
    "Research Scientist",
]
VALID_CITIES = [
    "New York", "San Francisco", "Chicago", "Austin", "Seattle",
    "Boston", "Denver", "Atlanta", "Remote",
]


class InvalidInputError(Exception):
    """Raised when user-supplied prediction inputs fail validation."""


def _load_model(model_path: str = MODEL_PATH):
    """Load the trained pipeline (preprocessor + regressor) from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run `python -m src.train` first to train and save a model."
        )
    return joblib.load(model_path)


def validate_inputs(age, gender, education, experience, job_title, city):
    """
    Validate raw prediction inputs before they reach the model.

    Raises
    ------
    InvalidInputError
        If any input is out of the expected range/type/category.
    """
    errors = []

    try:
        age = float(age)
        if not (16 <= age <= 100):
            errors.append("Age must be between 16 and 100.")
    except (TypeError, ValueError):
        errors.append("Age must be a number.")

    try:
        experience = float(experience)
        if experience < 0 or experience > 60:
            errors.append("Years of Experience must be between 0 and 60.")
    except (TypeError, ValueError):
        errors.append("Years of Experience must be a number.")

    if gender not in VALID_GENDERS:
        errors.append(f"Gender must be one of {VALID_GENDERS}.")

    if education not in VALID_EDUCATION_LEVELS:
        errors.append(f"Education Level must be one of {VALID_EDUCATION_LEVELS}.")

    if job_title not in VALID_JOB_TITLES:
        errors.append(f"Job Title must be one of {VALID_JOB_TITLES}.")

    if city not in VALID_CITIES:
        errors.append(f"City must be one of {VALID_CITIES}.")

    if isinstance(age, (int, float)) and isinstance(experience, (int, float)):
        if experience > (age - 16):
            errors.append(
                "Years of Experience is not realistic given the provided Age."
            )

    if errors:
        raise InvalidInputError(" | ".join(errors))

    return age, experience


def predict_salary(age, gender, education, experience, job_title, city,
                    model_path: str = MODEL_PATH) -> float:
    """
    Predict salary for a single individual.

    Parameters
    ----------
    age : int or float
    gender : str
        One of VALID_GENDERS.
    education : str
        One of VALID_EDUCATION_LEVELS.
    experience : int or float
        Years of professional experience.
    job_title : str
        One of VALID_JOB_TITLES.
    city : str
        One of VALID_CITIES.
    model_path : str
        Path to the saved model pipeline.

    Returns
    -------
    float
        Predicted salary.

    Raises
    ------
    InvalidInputError
        If inputs fail validation.
    FileNotFoundError
        If no trained model exists yet.
    """
    age, experience = validate_inputs(age, gender, education, experience, job_title, city)

    pipeline = _load_model(model_path)

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience,
        "City": city,
    }])[ALL_FEATURES]

    try:
        prediction = pipeline.predict(input_df)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    prediction = float(round(prediction, 2))
    logger.info(
        "Predicted salary=%.2f for input=%s", prediction, input_df.iloc[0].to_dict()
    )
    return prediction


def _prompt(prompt_text: str, default: str = None) -> str:
    """Small helper for interactive CLI prompting."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    return value if value else default


def interactive_cli():
    """Run an interactive command-line prediction session."""
    print("=" * 60)
    print("  SALARY PREDICTION - Command Line Interface")
    print("=" * 60)
    print(f"Valid genders: {VALID_GENDERS}")
    print(f"Valid education levels: {VALID_EDUCATION_LEVELS}")
    print(f"Valid job titles: {VALID_JOB_TITLES}")
    print(f"Valid cities: {VALID_CITIES}")
    print("-" * 60)

    try:
        age = _prompt("Age", "30")
        gender = _prompt("Gender", "Male")
        education = _prompt("Education Level", "Bachelor's")
        experience = _prompt("Years of Experience", "5")
        job_title = _prompt("Job Title", "Data Scientist")
        city = _prompt("City", "Remote")

        salary = predict_salary(age, gender, education, experience, job_title, city)
        print("\n" + "=" * 60)
        print(f"  PREDICTED SALARY: ${salary:,.2f}")
        print("=" * 60)

    except InvalidInputError as e:
        print(f"\n[Invalid Input] {e}")
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
    except Exception as e:
        logger.exception("Unexpected error during CLI prediction")
        print(f"\n[Unexpected Error] {e}")


if __name__ == "__main__":
    interactive_cli()
