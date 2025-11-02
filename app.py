import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
feature_columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction App")
st.write("Fill in customer details to predict churn:")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 100)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

if st.button("Predict Churn"):
    input_dict = {
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "InternetService": [InternetService],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
    }

    df_input = pd.DataFrame(input_dict)

    for col, encoder in encoders.items():
        if col in df_input.columns:
            df_input[col] = encoder.transform(df_input[col])

    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn ({prob:.2%} probability)")
    else:
        st.success(f"Customer likely to stay ({1 - prob:.2%} probability)")



