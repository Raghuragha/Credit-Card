import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load the trained model
# -----------------------------
filename = r'logistic_regression_model.joblib'
loaded_model = joblib.load(open(filename, 'rb'))

# Get the features the model was trained on
if hasattr(loaded_model, "feature_names_in_"):
    expected_features = list(loaded_model.feature_names_in_)
else:
    # Fallback (edit this if your model has different features)
    expected_features = [
        'Age', 'AnnualIncome', 'SpendingScore',
        'Tenure', 'CreditScore',
        'MaritalStatus_Married', 'MaritalStatus_Single'
    ]

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Credit Card Eligibility Prediction")

st.write("Please provide the following information:")

# Numeric inputs
Age = st.number_input("Age (18+)", min_value=18.0, step=1.0)
AnnualIncome = st.number_input("Annual Income", min_value=0.0, step=1000.0)
SpendingScore = st.number_input("Spending Score (1–100)", min_value=1.0, max_value=100.0, step=1.0)
Tenure = st.number_input("Tenure (years)", min_value=0, step=1)
CreditScore = st.number_input("Credit Score", min_value=0.0, step=10.0)

# Categorical input
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])

# -----------------------------
# Build input DataFrame
# -----------------------------
# Step 1: raw DataFrame
input_raw = pd.DataFrame([{
    'Age': Age,
    'AnnualIncome': AnnualIncome,
    'SpendingScore': SpendingScore,
    'Tenure': Tenure,
    'CreditScore': CreditScore,
    'MaritalStatus': MaritalStatus
}])

# Step 2: one-hot encode MaritalStatus
input_processed = pd.get_dummies(input_raw, columns=['MaritalStatus'])

# Step 3: add any missing expected columns with 0
for col in expected_features:
    if col not in input_processed.columns:
        input_processed[col] = 0

# Step 4: reorder columns to match model
input_processed = input_processed[expected_features]

# -----------------------------
# Debug info
# -----------------------------
st.write("### Model expected features:")
st.write(expected_features)

st.write("### Processed Input to Model:")
st.dataframe(input_processed)

# Show coefficients if available (e.g. LogisticRegression)
if hasattr(loaded_model, "coef_") and hasattr(loaded_model, "classes_") and hasattr(loaded_model, "feature_names_in_"):
    try:
        coef_df = pd.DataFrame({
            "Feature": loaded_model.feature_names_in_,
            "Coefficient": loaded_model.coef_[0]
        })
        st.write("### Model Coefficients (log-odds impact):")
        st.dataframe(coef_df)
    except Exception:
        st.write("Model coefficients not available for display.")

# Threshold slider
st.write("### Decision Threshold (for probability of approval)")
threshold = st.slider("Approval Threshold", 0.0, 1.0, 0.5, 0.01)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Eligibility"):
    try:
        # Probability of class 1 (Approved) if available
        if hasattr(loaded_model, "predict_proba"):
            proba = loaded_model.predict_proba(input_processed)[0][1]  # probability of class 1
            st.write(f"Approval Probability (class 1): **{proba:.4f}**")

            if proba >= threshold:
                st.success("✅ Predicted Eligibility: Approved")
            else:
                st.error("❌ Predicted Eligibility: Not Approved")
        else:
            # Fallback to plain predict
            prediction = loaded_model.predict(input_processed)
            if prediction[0] == 1:
                st.success("✅ Predicted Eligibility: Approved")
            else:
                st.error("❌ Predicted Eligibility: Not Approved")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
