import sys
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# so we can import from the common/ folder one level up
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.preprocessing import transform_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorizer.pkl")

df = pd.read_csv(DATA_PATH)

# clean every email the same way the API will clean incoming ones later
df["cleaned"] = df["text"].astype(str).apply(transform_text)

X = df["cleaned"]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# turn text into numeric features the model can actually learn from
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=["ham", "spam"]))

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"saved model to {MODEL_PATH}")
print(f"saved vectorizer to {VECTORIZER_PATH}")
