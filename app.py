"""
app.py
------
Streamlit web application for the Salary Prediction using Ensemble
Learning project.

Features:
    - Professional title and layout
    - Sidebar inputs for Age, Gender, Education, Experience, Job Title, City
    - Predict button that calls the saved model via src/predict.py
    - Displays the predicted salary prominently
    - Shows a feature importance chart from the trained model
    - Displays model comparison metrics (if available)
    - Graceful handling of invalid inputs / missing model

Run:
    streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.evaluate import get_feature_importance
from src.predict import (
    InvalidInputError,
    VALID_CITIES,
    VALID_EDUCATION_LEVELS,
    VALID_GENDERS,
    VALID_JOB_TITLES,
    predict_salary,
)
from src.utils import METRICS_PATH, MODEL_PATH

sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction | Ensemble Learning",
    page_icon="💼",
    layout="wide",
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("💼 Salary Prediction using Ensemble Learning")
st.markdown(
    """
    This application predicts an employee's expected annual salary using an
    ensemble machine learning model (Random Forest / Gradient Boosting /
    Extra Trees / XGBoost) trained on Age, Gender, Education Level,
    Years of Experience, Job Title, and City.
    """
)
st.divider()

# ----------------------------------------------------------------------
# Sidebar - user inputs
# ----------------------------------------------------------------------
st.sidebar.header("🧾 Candidate Details")
st.sidebar.markdown("Fill in the details below and click **Predict Salary**.")

age = st.sidebar.number_input(
    "Age", min_value=16, max_value=100, value=30, step=1,
    help="Age of the individual in years."
)
gender = st.sidebar.selectbox("Gender", options=VALID_GENDERS)
education = st.sidebar.selectbox("Education Level", options=VALID_EDUCATION_LEVELS)
experience = st.sidebar.number_input(
    "Years of Experience", min_value=0.0, max_value=60.0, value=5.0, step=0.5,
    help="Total years of professional work experience."
)
job_title = st.sidebar.selectbox("Job Title", options=VALID_JOB_TITLES)
city = st.sidebar.selectbox("City", options=VALID_CITIES)

predict_clicked = st.sidebar.button("🔮 Predict Salary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Model file: `models/salary_model.pkl`\n\n"
    "If this file doesn't exist yet, run `python main.py` or "
    "`python -m src.train` first."
)

# ----------------------------------------------------------------------
# Main area - layout with two columns
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("📊 Prediction Result")

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found. Please train the model first by running "
            "`python main.py` or `python -m src.train` in your terminal, "
            "then refresh this page."
        )
    elif predict_clicked:
        try:
            salary = predict_salary(
                age=age,
                gender=gender,
                education=education,
                experience=experience,
                job_title=job_title,
                city=city,
            )
            st.success("Prediction successful!")
            st.metric(label="Predicted Annual Salary", value=f"${salary:,.2f}")

            with st.expander("View input summary"):
                st.json({
                    "Age": age,
                    "Gender": gender,
                    "Education Level": education,
                    "Years of Experience": experience,
                    "Job Title": job_title,
                    "City": city,
                })

        except InvalidInputError as e:
            st.error(f"Invalid input: {e}")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001 - surfaced to the user intentionally
            st.error(f"Something went wrong while predicting: {e}")
    else:
        st.info("Enter candidate details in the sidebar and click **Predict Salary**.")

with col_right:
    st.subheader("🌟 Feature Importance")

    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            importance_df = get_feature_importance(pipeline).head(10)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=importance_df, x="Importance", y="Feature", hue="Feature",
                palette="mako", legend=False, ax=ax,
            )
            ax.set_title("Top 10 Feature Importances")
            st.pyplot(fig)
        except AttributeError:
            st.info("The current model type does not expose feature importances.")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not compute feature importance: {e}")
    else:
        st.info("Train the model first to see feature importance.")

st.divider()

# ----------------------------------------------------------------------
# Model comparison table (if available)
# ----------------------------------------------------------------------
st.subheader("📈 Model Comparison (from training)")

if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=metrics_df, x="Model", y="R2 Score", hue="Model",
        palette="viridis", legend=False, ax=ax2,
    )
    ax2.set_ylim(0, 1)
    ax2.set_title("R2 Score by Model")
    plt.xticks(rotation=15)
    st.pyplot(fig2)
else:
    st.info(
        "No comparison table found yet. Run `python -m src.train` to "
        "generate model comparison results."
    )

st.divider()
st.caption(
    "Built with scikit-learn, XGBoost (optional), and Streamlit • "
    "Salary Prediction using Ensemble Learning"
)
