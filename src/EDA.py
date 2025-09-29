import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load churn dataset
df = pd.read_csv (r'data/telco_churn.csv')

# Plot churn distribution
sns.countplot(x="Churn", data=df, palette="Set2")
plt.title("Churn Distribution")
plt.show()

# Compare MonthlyCharges for churned vs non-churned customers
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set1")
plt.title("Monthly Charges vs Churn")
plt.show()
