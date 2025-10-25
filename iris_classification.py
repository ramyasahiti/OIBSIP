# ────────────────────────────────────────────────
# IRIS FLOWER CLASSIFICATION 
# ────────────────────────────────────────────────

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\Oasis\Iris.csv")


# Display first few rows
print("Sample data:\n", df.head(), "\n")

# 2. Prepare features and labels
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# 3. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Support Vector Machine)": SVC(kernel='rbf', random_state=42)
}

# 6. Train & Evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    results.append([name, acc, prec, rec, f1])

# 7. Display results as DataFrame
df_results = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"])
print("\nModel Performance (in %):\n")
print(df_results.to_string(index=False))

# 8. Visualize results
plt.figure(figsize=(8,5))
plt.barh(df_results["Model"], df_results["Accuracy (%)"], color='skyblue')
plt.xlabel("Accuracy (%)")
plt.title("Model Comparison - Iris Flower Classification")
plt.xlim(90, 101)
plt.show()
