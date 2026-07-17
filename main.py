import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.keras")
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.title("Customer Churn Prediction")

# User Inputs
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=50000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", [1, 0])
IsActiveMember = st.selectbox("Is Active Member", [1, 0])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
Geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)
# Prediction
if st.button("Predict"):

    # Create DataFrame
    df = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary],
        "Geography": [Geography],
        "Gender": [Gender]
    })
    X = preprocessor.transform(df)
    prediction = model.predict(X)

    if prediction[0][0] >= 0.45:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")