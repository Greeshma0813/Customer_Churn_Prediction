import streamlit as st
import pandas as pd
import pickle
import os

# ------------------ Paths ------------------
MODEL_PATH = r'model/model.pkl'

# ------------------ Load Model ------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please run train_model.py first to create model.pkl")
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

# ------------------ User Inputs ------------------
def user_input_features():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', 0, 100, 12)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', 0.0, 1000.0, 70.0)
    total_charges = st.number_input('Total Charges', 0.0, 10000.0, 800.0)

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# ------------------ Prediction ------------------
if st.button('Predict Churn'):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        if prediction[0] == 1:
            st.error(f"The customer is likely to churn. Probability: {prediction_proba[0]:.2f}")
        else:
            st.success(f"The customer is unlikely to churn. Probability: {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
