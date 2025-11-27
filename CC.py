import streamlit as st
import pandas as pd
import joblib

# Load the trained model
filename = r'logistic_regression_model.joblib'
loaded_model = joblib.load(open(filename, 'rb'))

# Define the correct column names
columns = ['Age', 'AnnualIncome', 'SpendingScore','Tenure', 'CreditScore'],

# Define the prediction function
def predict_Eligibility(features):
    """
    Predicts the delivery delay based on input features.
    """
    prediction = loaded_model.predict(features)
    return prediction

# Create the Streamlit app
st.title("Credict card Eligibility Prediction")

# Get user input
st.write("Please provide the following information:")
Age = st.number_input("Age(18+)", min_value=0.0)
AnnualIncome = st.number_input("Annual Income", min_value=0.0)
SpendingScore = st.number_input("Spending Score (1-100)", min_value=0.0)
Tenure = st.number_input("Tenure", min_value=1)
CreditScore = st.number_input("Credict Score", min_value=0.0)

# Create a dataframe with the user input
input_data = pd.DataFrame([[Age, AnnualIncome, SpendingScore, Tenure,CreditScore]], columns=columns)

# Make a prediction
# Make a prediction
if st.button("Predict Eligibility"):
    prediction = predict_Eligibility(input_data)
    if prediction[0] == 0:
        st.write("Predicted Eligibility: 0 (Not Approved)")
    else:
        st.write("Predicted Eligibility: 1 (Approved)")
