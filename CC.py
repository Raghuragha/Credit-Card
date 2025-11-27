import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load the trained model
# -----------------------------
filename = r'logistic_regression_model.joblib'
loaded_model = joblib.load(open(filename, 'rb'))

# Get the features the model was trained on
# (this will include MaritalStatus_Married, MaritalStatus_Single, etc.)
if hasattr(loaded_model, "feature_names_in_"):
    expected_features = list(loaded_model.feature_names_in_)
else:
    # Fallback (only if model was trained without feature names)
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
# Step 1: Create base DataFrame with raw values
input_raw = pd.DataFrame([{
    'Age': Age,
    'AnnualIncome': AnnualIncome,
    'SpendingScore': SpendingScore,
    'Tenure': Tenure,
    'CreditScore': CreditScore,
    'MaritalStatus': MaritalStatus
}])

# Step 2: One-hot encode MaritalStatus to match training
input_processed = pd.get_dummies(input_raw, columns=['MaritalStatus'])

# Step 3: Ensure all expected model features exist; if missing, fill with 0
for col in expected_features:
    if col not in input_processed.columns:
        input_processed[col] = 0

# Step 4: Reorder columns to match the model’s training order
input_processed = input_processed[expected_features]

# Show processed input (for debugging / transparency)
st.write("### Processed Input to Model")
st.dataframe(input_processed)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Eligibility"):
    try:
        prediction = loaded_model.predict(input_processed)

        if prediction[0] == 0:
            st.error("❌ Predicted Eligibility: Not Approved")
        else:
            st.success("✅ Predicted Eligibility: Approved")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
