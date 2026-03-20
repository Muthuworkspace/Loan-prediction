import streamlit as st
import pandas as pd
import joblib

model = joblib.load('loan_model_v1.pkl')

st.title("Bank Loan Prediction System")

income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
credit = st.selectbox("Credit History", [0, 1])


if st.button("Predict"):
  
    data = pd.DataFrame([[income, co_income, credit]], 
                        columns=['ApplicantIncome', 'CoapplicantIncome', 'Credit_History'])

    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.success("Loan Approved! ✅")
    else:
        st.error("Loan Rejected! ❌")