import pandas as pd
import re
import nltk
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

BASE_DIR = Path(__file__).resolve().parent   # anchored to script location
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Load data
data1 = pd.read_csv(DATA_DIR / "Fake.csv")
data2 = pd.read_csv(DATA_DIR / "True.csv")
data1["label"] = 1
data2["label"] = 0

df = pd.concat([data1, data2]).sample(frac=1).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]

# Text cleaning
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_content"] = df["content"].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_content"], df["label"], test_size=0.2, random_state=48
)

# TF-IDF + SVM
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save to /models
joblib.dump(model, MODELS_DIR / "fake_news_svm_model.pkl")
joblib.dump(tfidf,  MODELS_DIR / "tfidf_vectorizer.pkl")
print("Model and vectorizer saved to /models/")