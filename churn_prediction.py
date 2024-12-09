# Churn Prediction with Updated Visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
Households5 = pd.read_csv('400_households.csv')
Products = pd.read_csv('400_products.csv')
Transactions3 = pd.read_csv('400_transactions.csv')

# Set ggplot style
plt.style.use('ggplot')

# Updated Color Palette
colors = sns.color_palette("coolwarm", as_cmap=False)

# Clean column names
Households5.columns = Households5.columns.str.strip()
Products.columns = Products.columns.str.strip()
Transactions3.columns = Transactions3.columns.str.strip()

# Merge datasets
data = Transactions3.merge(Products, on="PRODUCT_NUM", how="left")
data = data.merge(Households5, on="HSHD_NUM", how="left")

# Convert 'PURCHASE_' column to datetime
data['PURCHASE_'] = pd.to_datetime(data['PURCHASE_'], errors='coerce')

# Get the most recent purchase date for each household
data['last_purchase'] = data.groupby('HSHD_NUM')['PURCHASE_'].transform('max')

# Convert 'last_purchase' to datetime if not already
data['last_purchase'] = pd.to_datetime(data['last_purchase'], errors='coerce')

# Create a churn label: customers who have not made a purchase in the last 90 days are considered to have churned
threshold_date = pd.to_datetime('2024-01-01')  # Use the appropriate threshold date here
data['churn'] = (data['last_purchase'] < threshold_date).astype(int)

# Convert datetime columns to numeric (e.g., days since the first purchase)
data['last_purchase_numeric'] = (data['last_purchase'] - data['last_purchase'].min()).dt.days

# Drop non-numeric columns
data = data.drop(columns=['HSHD_NUM', 'last_purchase', 'PURCHASE_', 'STORE_R'])

# One-Hot Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data.drop(columns=['churn'])
y = data['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance with Updated Look
plt.figure(figsize=(10, 6))
plt.barh(
    importance['Feature'][:10], importance['Importance'][:10],
    color=colors[:10],
    edgecolor='white'
)
plt.xlabel('Importance', fontsize=14, color='black')
plt.ylabel('Feature', fontsize=14, color='black')
plt.title('Top Features for Churn Prediction', fontsize=16, color='black')
plt.gca().invert_yaxis()

# Add gridlines and adjust layout
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the chart
plt.savefig('static/feature_importance_churn_updated.png', dpi=300, bbox_inches="tight")
print("Feature Importance chart saved.")

# Save model accuracy
with open('static/model_accuracy_churn.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.2f}")
