"""
main.py
-------
Single entry point that runs the entire Salary Prediction pipeline
end-to-end:
    1. Generate the dataset (if it doesn't already exist)
    2. Run EDA and save charts
    3. Preprocess data, train & compare models, tune the best one, save it
    4. Run a sample prediction to confirm everything works

Usage:
    python main.py
"""

import os
import subprocess
import sys

from src.utils import DATASET_PATH, MODEL_PATH, get_logger

logger = get_logger(__name__)


def step_generate_dataset():
    if os.path.exists(DATASET_PATH):
        logger.info("Dataset already exists at %s - skipping generation.", DATASET_PATH)
        return
    logger.info("Dataset not found - generating synthetic dataset ...")
    subprocess.run([sys.executable, "dataset/generate_dataset.py"], check=True)


def step_run_eda():
    logger.info("Running Exploratory Data Analysis ...")
    subprocess.run([sys.executable, "notebooks/eda.py"], check=True)


def step_train_model():
    logger.info("Training models ...")
    from src.train import main as train_main
    train_main()


def step_sample_prediction():
    logger.info("Running a sample prediction to verify the saved model ...")
    from src.predict import predict_salary

    salary = predict_salary(
        age=29,
        gender="Female",
        education="Master's",
        experience=5,
        job_title="Data Scientist",
        city="Seattle",
    )
    print(f"\nSample prediction -> Predicted Salary: ${salary:,.2f}\n")


def main():
    print("=" * 70)
    print(" SALARY PREDICTION USING ENSEMBLE LEARNING - Full Pipeline")
    print("=" * 70)

    step_generate_dataset()
    step_run_eda()
    step_train_model()
    step_sample_prediction()

    print("=" * 70)
    print(" Pipeline complete!")
    print(f" Trained model saved at: {MODEL_PATH}")
    print(" Launch the web app with: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
