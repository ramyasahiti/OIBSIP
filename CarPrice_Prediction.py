# TASK 2 - CAR PRICE PREDICTION
# Oasis Infobyte Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Step 1: Load Dataset
df = pd.read_csv("car data.csv")

print("Original Columns:\n", df.columns, "\n")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
print("Cleaned Columns:\n", df.columns, "\n")

# Step 2: Inspect Data
print("First 5 rows:")
print(df.head(), "\n")

print("Dataset Info:")
print(df.info(), "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Step 3: Encode Categorical Columns
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Encoded categorical columns:", cat_cols, "\n")

# Step 4: Define Features and Target
if "selling_price" in df.columns:
    y = df["selling_price"]
else:
    raise KeyError("Couldn't find the 'Selling_Price' column in dataset!")

# Drop irrelevant/non-numeric columns like car_name
X = df.drop(columns=["car_name", "selling_price"], errors="ignore")

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train and Evaluate Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = [r2, mae, rmse]
    print(f"===== {name} =====")
    print("R² Score:", round(r2, 3))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print()

# Step 8: Compare Model Performance
results_df = pd.DataFrame(results, index=["R2 Score", "MAE", "RMSE"]).T
print("Model Performance Comparison:\n")
print(results_df, "\n")

# Step 9: Visualization
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Selling Price (in lakhs)")
plt.ylabel("Predicted Selling Price (in lakhs)")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x=results_df.index, y=results_df["R2 Score"], palette="viridis")
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.show()

# Step 10: Final Summary
print("Best Performing Model:", results_df["R2 Score"].idxmax())
print("Highest R² Score:", round(results_df["R2 Score"].max(), 3))

