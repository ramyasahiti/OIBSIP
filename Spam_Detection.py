# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Encode Labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# Step 5: Vectorize Text
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test_cv)

# Step 8: Evaluate Model
accuracy = accuracy_score(y_test, y_pred) * 100  # convert to percentage
print(f"Accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Test on Custom Email
def predict_email(text):
    text_cv = cv.transform([text])
    prediction = model.predict(text_cv)[0]
    return "SPAM" if prediction == 1 else "NOT SPAM"

# Example
sample = "Congratulations! You’ve won a free iPhone. Click the link to claim now."
print("\nSample Email Prediction:", predict_email(sample))

