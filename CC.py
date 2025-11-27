import streamlit as st
import pandas as pd
import joblib

# Load the trained model
filename = r'logistic_regression_model.joblib'
loaded_model = joblib.load(open(filename, 'rb'))

# ✅ Define the correct column names (no trailing comma)
columns = ['Age', 'AnnualIncome', 'SpendingScore', 'Tenure', 'CreditScore']

# Define the prediction function
def predict_Eligibility(features):
    """
    Predicts credit card eligibility based on input features.
    """
    prediction = loaded_model.predict(features)
    return prediction

# Create the Streamlit app
st.title("Credit Card Eligibility Prediction")

# Get user input
st.write("Please provide the following information:")

Age = st.number_input("Age (18+)", min_value=18.0, step=1.0)
AnnualIncome = st.number_input("Annual Income", min_value=0.0, step=1000.0)
SpendingScore = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, step=1.0)
Tenure = st.number_input("Tenure (years with the company)", min_value=0, step=1)
CreditScore = st.number_input("Credit Score", min_value=0.0, step=10.0)

# ✅ Create a DataFrame with the user input
input_data = pd.DataFrame([[Age, AnnualIncome, SpendingScore, Tenure, CreditScore]], columns=columns)

# Display the input data (optional)
st.write("### Input Summary")
st.dataframe(input_data)

# Make a prediction
if st.button("Predict Eligibility"):
    prediction = predict_Eligibility(input_data)
    if prediction[0] == 0:
        st.error("❌ Predicted Eligibility: Not Approved")
    else:
        st.success("✅ Predicted Eligibility: Approved")
