"""
generate_dataset.py
--------------------
Generates a realistic, synthetic "Salary Dataset" (salary.csv) for the
Salary Prediction using Ensemble Learning project.

WHY A GENERATED DATASET?
This project is designed to work with the well-known public
"Salary Data" dataset (commonly found on Kaggle, e.g. search for
"Salary Data - Years of Experience, Age, Gender, Education, Job Title").
That dataset has the following columns:
    Age, Gender, Education Level, Job Title, Years of Experience, Salary

Because this development environment has no internet access, this script
generates a synthetic dataset with the SAME schema (plus a bonus "City"
column, as the assignment permits City "if available"). The generation
logic encodes realistic salary relationships (experience, education,
age, job seniority, and city cost-of-living all influence salary, with
random noise) so that every downstream script (EDA, preprocessing,
training, evaluation) behaves exactly as it would on real-world data.

TO USE A REAL DATASET INSTEAD:
Simply download a real salary CSV (e.g. from Kaggle) with matching
column names and replace `dataset/salary.csv`. No other code changes
are required as long as the column names match.

Run:
    python dataset/generate_dataset.py
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N_SAMPLES = 2000

# ----------------------------------------------------------------------
# Feature definitions
# ----------------------------------------------------------------------
genders = ["Male", "Female"]

education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
education_salary_boost = {
    "High School": 0,
    "Bachelor's": 15000,
    "Master's": 30000,
    "PhD": 50000,
}

job_titles = [
    "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
    "HR Manager", "Sales Executive", "Marketing Manager", "Business Analyst",
    "Financial Analyst", "Operations Manager", "Project Manager",
    "UX Designer", "DevOps Engineer", "Customer Support Specialist",
    "Research Scientist",
]
job_title_salary_boost = {
    "Software Engineer": 20000,
    "Data Analyst": 8000,
    "Data Scientist": 28000,
    "Product Manager": 30000,
    "HR Manager": 10000,
    "Sales Executive": 5000,
    "Marketing Manager": 15000,
    "Business Analyst": 10000,
    "Financial Analyst": 12000,
    "Operations Manager": 18000,
    "Project Manager": 22000,
    "UX Designer": 14000,
    "DevOps Engineer": 24000,
    "Customer Support Specialist": 2000,
    "Research Scientist": 26000,
}

cities = ["New York", "San Francisco", "Chicago", "Austin", "Seattle",
          "Boston", "Denver", "Atlanta", "Remote"]
city_cost_of_living_boost = {
    "New York": 18000,
    "San Francisco": 25000,
    "Chicago": 8000,
    "Austin": 6000,
    "Seattle": 15000,
    "Boston": 12000,
    "Denver": 5000,
    "Atlanta": 4000,
    "Remote": 0,
}

# ----------------------------------------------------------------------
# Generate correlated, realistic features
# ----------------------------------------------------------------------
records = []
for _ in range(N_SAMPLES):
    age = int(np.clip(np.random.normal(35, 9), 21, 65))
    gender = np.random.choice(genders, p=[0.55, 0.45])
    education = np.random.choice(
        education_levels, p=[0.15, 0.45, 0.30, 0.10]
    )

    # Years of experience is bounded by age and correlated with it
    max_possible_exp = max(age - 21, 0)
    years_experience = round(
        np.clip(np.random.normal(max_possible_exp * 0.55, 3), 0, max_possible_exp), 1
    )

    job_title = np.random.choice(job_titles)
    city = np.random.choice(cities)

    # Base salary formula (realistic, with multiple contributing factors)
    base_salary = 30000
    salary = (
        base_salary
        + years_experience * 2200
        + education_salary_boost[education]
        + job_title_salary_boost[job_title]
        + city_cost_of_living_boost[city]
        + (age - 21) * 150
        + np.random.normal(0, 7000)  # random noise
    )
    salary = round(max(salary, 20000), 2)  # floor salary, avoid negatives

    records.append(
        {
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": years_experience,
            "City": city,
            "Salary": salary,
        }
    )

df = pd.DataFrame(records)

# ----------------------------------------------------------------------
# Inject a small amount of realistic "messiness" so preprocessing code
# (missing values, duplicates) has something genuine to clean.
# ----------------------------------------------------------------------
# 1. Randomly introduce missing values (~2% of cells in a few columns)
for col in ["Age", "Gender", "Education Level", "Years of Experience"]:
    missing_idx = df.sample(frac=0.02, random_state=np.random.randint(0, 1000)).index
    df.loc[missing_idx, col] = np.nan

# 2. Duplicate a handful of rows
duplicate_rows = df.sample(n=20, random_state=1)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 3. Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
output_path = "dataset/salary.csv"
df.to_csv(output_path, index=False)
print(f"Synthetic salary dataset generated: {output_path}")
print(f"Shape: {df.shape}")
print(df.head())
