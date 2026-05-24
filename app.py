import streamlit as st
import joblib
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

MODELS_DIR = Path("models")

@st.cache_resource   # loads once, cached across reruns
def load_model():
    model = joblib.load(MODELS_DIR / "fake_news_svm_model.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, tfidf

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")
st.title("Fake News Detection App")
st.write("Paste a news article below and click **Predict**.")

model, tfidf = load_model()

news_text = st.text_area("News text:", height=200)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned   = clean_text(news_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error(" This news is **FAKE**")
        else:
            st.success("This news is **REAL**")