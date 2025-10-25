# Task 4 - Sales Prediction using Python (Oasis Infobyte Internship)
# Predicts future sales based on advertising budgets using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("Advertising.csv")


print("Sample Data:\n", df.head(), "\n")

# Define features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = r2_score(y_test, y_pred) * 100

# Display results
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("\nActual vs Predicted Sales:\n", pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head())
print(f"\nModel Accuracy: {accuracy:.2f}%")
