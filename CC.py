import streamlit as st
import pandas as pd
import joblib

# Load the trained model
filename = r'logistic_regression_model.joblib'
loaded_model = joblib.load(open(filename, 'rb'))

# Define the correct column names (as used during training)
columns = ['Age', 'AnnualIncome', 'SpendingScore', 'Tenure', 'CreditScore']

# Define the prediction function
def predict_Eligibility(features_df: pd.DataFrame):
    """
    Predicts credit card eligibility based on input features.
    Handles both:
      - models trained with feature names
      - models trained without feature names
    """
    # If the model was trained with feature names, align to them
    if hasattr(loaded_model, "feature_names_in_"):
        model_cols = list(loaded_model.feature_names_in_)
        # Reorder / subset the input to match training columns
        X = features_df[model_cols]
    else:
        # Model trained without feature names -> use raw values (NumPy array)
        X = features_df.values

    prediction = loaded_model.predict(X)
    return prediction

# Create the Streamlit app
st.title("Credit Card Eligibility Prediction")

# Get user input
st.write("Please provide the following information:")

Age = st.number_input("Age (18+)", min_value=18.0, step=1.0)
AnnualIncome = st.number_input("Annual Income", min_value=0.0, step=1000.0)
SpendingScore = st.number_input("Spending Score (1–100)", min_value=1.0, max_value=100.0, step=1.0)
Tenure = st.number_input("Tenure (years)", min_value=0, step=1)
CreditScore = st.number_input("Credit Score", min_value=0.0, step=10.0)

# Create a dataframe with the user input
input_data = pd.DataFrame(
    [[Age, AnnualIncome, SpendingScore, Tenure, CreditScore]],
    columns=columns
)

# Show the input for debugging / confirmation
st.write("### Input Summary")
st.dataframe(input_data)

# Make a prediction
if st.button("Predict Eligibility"):
    try:
        prediction = predict_Eligibility(input_data)
        if prediction[0] == 0:
            st.error("❌ Predicted Eligibility: Not Approved")
        else:
            st.success("✅ Predicted Eligibility: Approved")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
