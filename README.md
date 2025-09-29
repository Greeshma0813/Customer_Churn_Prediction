Customer Churn Prediction
Project Overview

This project predicts whether a customer will churn (leave a service) using machine learning. It uses a Random Forest classifier and handles both numeric and categorical features. The project includes a Streamlit web app where users can input customer details and get a churn prediction along with probability.

Project Structure
Customer_Churn_Prediction/
│
├── data/
│   └── telco_churn.csv           # Dataset file
│
├── model/
│   └── model.pkl                 # Trained ML pipeline (generated automatically)
│
├── src/
│   ├── train_model.py            # Script to train and save model
│   └── streamlit_app.py          # Streamlit web application
│
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation

Requirements

Python 3.8+

Packages listed in requirements.txt

Install dependencies:

pip install -r requirements.txt

Dataset

Dataset used: telco_churn.csv

Columns include customer info (gender, tenure, services, charges, etc.) and the target Churn.

Note: The customerID column is ignored during training.

How to Run
1. Train the Model
python src\train_model.py


This will preprocess data, train a Random Forest pipeline, and save it as model.pkl.

Automatically handles missing values in TotalCharges and categorical encoding.

2. Run the Streamlit App
streamlit run src\streamlit_app.py


Enter customer details in the web form.

Click Predict Churn to see whether the customer is likely to churn and the probability.

Features Used

Numeric: tenure, MonthlyCharges, TotalCharges

Categorical: Gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

Notes

Blank or missing TotalCharges are automatically converted to 0.

customerID is ignored during training and prediction.

Streamlit warning missing ScriptRunContext can be ignored; always run the app with:

streamlit run src\streamlit_app.py

Author

Greeshma