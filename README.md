# Predictive Maintenance & Failure Detection System

A machine learning project that estimates machine failure risk using operating conditions such as temperature, rotational speed, torque, and tool wear.

## Project Overview

This project uses the AI4I 2020 Predictive Maintenance Dataset to explore machine failure patterns, train classification models, and build an interactive risk prediction application using Streamlit.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and categorical encoding
- Logistic Regression and Random Forest classification
- Handling class imbalance
- Failure-risk scoring with a configurable classification threshold
- Feature importance visualization
- Interactive Streamlit application

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Model Performance

The selected Random Forest model was evaluated using a 0.30 classification threshold.

| Metric | Result |
|---|---:|
| Accuracy | 98.0% |
| Failure Precision | 66.7% |
| Failure Recall | 64.7% |
| Failure F1-score | 65.7% |

These results are from the project's reported evaluation. Performance may vary on other data, and the dataset is synthetic.

## How It Works

1. Explore and prepare the dataset.
2. Train classification models to identify potential machine failures.
3. Select a decision threshold to balance precision and recall.
4. Review overall feature importance.
5. Enter machine operating conditions in the Streamlit application to obtain an estimated failure risk.

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the app from the project folder:

```bash
streamlit run app.py
```

The trained model and feature files must be generated using the notebook and placed in the expected project folder before launching the app.

## Limitations

- The project uses a synthetic dataset, not live industrial sensor data.
- Feature importance describes overall model reliance; it does not prove causation or explain an individual prediction.
- The displayed risk indicators are predefined rules and are separate from the model's prediction.
- This project is for educational and portfolio demonstration only, not for real-world safety decisions.

## Dataset

AI4I 2020 Predictive Maintenance Dataset, available through the UCI Machine Learning Repository.

