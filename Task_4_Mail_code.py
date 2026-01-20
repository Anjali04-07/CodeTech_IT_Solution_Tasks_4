import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# Remove missing rows
df = df.dropna()

# Convert labels to numeric
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Double-check
print(df.isnull().sum())

# Split data
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)  # ✅ Should work now

# Test model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict custom email
email = ["Congratulations! You won a free prize"]
email_vec = vectorizer.transform(email)
result = model.predict(email_vec)
print("Spam" if result[0] == 1 else "Not Spam")
