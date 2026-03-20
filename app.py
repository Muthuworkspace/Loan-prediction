import streamlit as st
import pandas as pd
import joblib

# 1. Load the "90% Accuracy" Brain
model = joblib.load('loan_model_v1.pkl')

st.title("🏦 Elite Loan Approval System")

# 2. Create ALL the features the model expects
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0)
    co_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amt = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Term (Days)", value=360)
    credit_hist = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# 3. Predict Button
if st.button("Calculate Eligibility"):
    # CREATE THE DATA FRAME (Columns MUST match your X_train exactly!)
    input_data = pd.DataFrame([[
        gender, married, dependents, education, self_emp, 
        applicant_income, co_income, loan_amt, loan_term, 
        credit_hist, property_area
    ]], columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area'
    ])
    
    # Feature Engineering (Remember we added 'Total_Income' to hit 90%!)
    input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']

    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Loan Approved! ✅")
    else:
        st.error("Loan Rejected! ❌")