import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

# Load dataset and preprocess
df = pd.read_csv(r'data/telco_churn.csv')

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes":1, "No":0})
X = pd.get_dummies(X, drop_first=True)

# Train model (or load a saved model)
model = RandomForestClassifier()
model.fit(X, y)

# Confusion Matrix
cm = confusion_matrix(y, model.predict(X))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_prob = model.predict_proba(X)[:,1]
fpr, tpr, _ = roc_curve(y, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Churn Prediction")
plt.legend()
plt.show()
