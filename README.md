# 🏦 Loan Approval Prediction

A Machine Learning web application built with Python and Streamlit that predicts whether a loan application is likely to be approved based on applicant information.

## Overview

This project uses a Random Forest Classifier trained on historical loan application data to predict loan approval status. The model analyzes applicant details such as income, credit history, education, employment status, and property area to generate predictions with confidence scores.

## Features

- Interactive Streamlit web application
- Real-time loan approval prediction
- Automated data preprocessing
- Missing value handling
- One-Hot Encoding for categorical features
- Random Forest Classification model
- Prediction confidence score
- Feature importance visualization

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

### Open in Browser

```text
http://localhost:8501
```

## Dataset

**File:** `dataset/LoanApprovalPrediction.csv`

**Records:** 598 loan applications

**Target Variable:** `Loan_Status` (Approved / Rejected)

## Model Details

- **Algorithm:** Random Forest Classifier
- **Features Used:** 11
- **Train/Test Split:** 70/30
- **Validation Accuracy:** ~80%

## Project Structure

```text
Loan_Approval_Prediction/
│
├── app.py
├── Loan_Approval_Prediction.ipynb
├── requirements.txt
├── README.md
├── loan_model.joblib
└── dataset/
    └── LoanApprovalPrediction.csv
```

## Usage

1. Enter applicant details in the sidebar.
2. Click **Predict Loan Status**.
3. View the prediction result and confidence score.
4. Explore feature importance to understand model decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

## Author

**Kaif Khan**

- GitHub: https://github.com/kaifcs
- LinkedIn: https://www.linkedin.com/in/kaif-khan-2805-2005-cs/

## License

This project is intended for educational and learning purposes.