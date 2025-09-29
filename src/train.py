import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# ------------------ Paths ------------------
DATA_PATH = r'data/telco_churn.csv'
MODEL_PATH = r'model/model.pkl'

# Create model folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ------------------ Load Dataset ------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place telco_churn.csv in data/ folder.")

data = pd.read_csv(DATA_PATH)

# ------------------ Fix TotalCharges blanks ------------------
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Convert non-numeric to NaN
data['TotalCharges'] = data['TotalCharges'].fillna(0)  # Fill NaN with 0

# ------------------ Prepare Features & Target ------------------

X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify numeric and categorical features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# ------------------ Preprocessing Pipeline ------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Full pipeline with RandomForest
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ------------------ Split & Train ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# ------------------ Save Pipeline ------------------
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"Model trained and saved successfully as {MODEL_PATH}!")

#Visualization

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r'data/telco_churn.csv')

# Drop ID column
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes":1, "No":0})

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance for Churn Prediction")
plt.show()
