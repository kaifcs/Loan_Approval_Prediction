import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = Path("loan_model.joblib")
LOAN_TERMS = [360.0, 180.0, 240.0, 300.0, 120.0, 84.0, 60.0, 36.0, 480.0]

NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]


def find_dataset() -> Path:
    """Locate dataset in project structure."""
    for path in [Path("dataset") / "LoanApprovalPrediction.csv",
                 Path("..") / "dataset" / "LoanApprovalPrediction.csv"]:
        if path.exists():
            return path
    raise FileNotFoundError("Dataset not found in dataset/ or ../dataset/")


def train_model() -> tuple[Pipeline, float]:
    """Train and save model. Returns (model, accuracy)."""
    df = pd.read_csv(find_dataset())
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Loan_Status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler())
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ])),
        ("classifier", RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=6
        ))
    ])
    
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    return model, accuracy


@st.cache_resource
def load_model() -> tuple[Pipeline, float]:
    """Load trained model or train new one."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), 0.0
    return train_model()


def app():
    st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
    st.title("Loan Approval Predictor")
    st.write(
        "Predict loan approval likelihood based on applicant profile. "
        "Model automatically handles preprocessing and returns confidence score."
    )

    try:
        model, accuracy = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if accuracy > 0:
        st.success(f"✓ Model trained with {accuracy*100:.1f}% validation accuracy")
    else:
        st.info("Loaded existing model")

    st.sidebar.header("Applicant Details")
    data = {
        "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "Married": st.sidebar.selectbox("Married", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"]),
        "Education": st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"]),
        "Self_Employed": st.sidebar.selectbox("Self Employed", ["Yes", "No"]),
        "Property_Area": st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"]),
        "ApplicantIncome": st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=500),
        "CoapplicantIncome": st.sidebar.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0),
        "LoanAmount": st.sidebar.number_input("Loan Amount", min_value=0.0, value=100.0, step=10.0),
        "Loan_Amount_Term": st.sidebar.selectbox("Loan Term (months)", LOAN_TERMS),
        "Credit_History": st.sidebar.selectbox("Credit History", [1.0, 0.0]),
    }

    if st.button("Predict Loan Status"):
        df_input = pd.DataFrame([data])
        pred_label = model.predict(df_input)[0]
        pred_prob = model.predict_proba(df_input)[0][int(pred_label)]
        
        status = "Approved" if pred_label == 1 else "Rejected"
        confidence = pred_prob * 100
        
        if status == "Approved":
            st.success(f"✅ Loan {status} ({confidence:.1f}% confidence)")
        else:
            st.error(f"❌ Loan {status} ({confidence:.1f}% confidence)")
        
        st.write("**Input Summary:**")
        st.json(data)

        with st.expander("See feature importance"):
            features = model.named_steps["preprocessor"].get_feature_names_out()
            importance = model.named_steps["classifier"].feature_importances_
            importance_df = pd.Series(importance, index=features).sort_values(ascending=False).head(10)
            st.bar_chart(importance_df)


if __name__ == "__main__":
    app()