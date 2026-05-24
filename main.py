import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# datasets fake
data1 = pd.read_csv(DATA_DIR / "Fake.csv")
# data1.head()

# real data
data2 = pd.read_csv(DATA_DIR / "True.csv")
# data2.head()

#information
# data1.info()
# data2.info()


#add column in fake csv as 1
data1["label"]=1
#add column in real csv as 0
data2["label"]=0


#combine all dataset
df = pd.concat([data1, data2], axis=0)
df = df.sample(frac=1).reset_index(drop=True)   # shuffle
# df.shape
# df.head()

#add new column(title and text)
df["content"] = df["title"]+" "+df["text"]



#using nlp library deep learning
import re
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer #sentence into words root form

#text cleaninig
# lowercase
# remove URLs
# remove punctuation
# remove numbers
# remove special characters
# remove stopwords
# stemming

stop_words= set(stopwords.words("english"))
stemmer=PorterStemmer()

def clean_text(text):
    text=text.lower() #lower case
    text=re.sub(r"http\S+|www\S+|https\S+", "", text) #remove urls
    #re.sub() means substitute / replace something in text.
#     re.sub(pattern, replace_with, text)
# So here:
# pattern = what you want to find
# replace_with = what you want to put instead
# text = original sentence
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation, numbers, special characters
    words = text.split()  # it breaks the sentence into seperate words
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # stopwords + stemming 
# Apply stemming
# Stemming means converting words into their root form.
# Example:
# "playing" → "play"
# "connected" → "connect"
# "studies" → "studi" (sometimes looks weird, but works)
# This part does stemming:
# stemmer.stem(word)
    return " ".join(words)


#apply cleaning 
df["clean_content"] = df["content"].apply(clean_text)

# print("before cleaninig : \n ",df["content"].iloc[0])
# print("after cleaninig : \n",df["clean_content"].iloc[0])

#evaluation 

#taking input and output as X,Y
X=df["clean_content"]
Y=df["label"]

#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=48)

#TF-IDF vectorization
#convert text into number

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


#train svm model
from sklearn.svm import LinearSVC
model=LinearSVC()

model.fit(X_train_tfidf, Y_train)


#predict the test data

y_pred = model.predict(X_test_tfidf)

#Evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print("Accuracy:", accuracy_score(Y_test, y_pred))
# print("\nClassification Report:\n", classification_report(Y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))

#save model + Td-idf (for streamlit)
import joblib
joblib.dump(model, MODELS_DIR / "fake_news_svm_model.pkl")
joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
# print("Model and Vectorizer Saved Successfully")


#test with my own news text
sample_news = ["Government announces new policy for students"]

sample_news_clean = [clean_text(sample_news[0])]
sample_vec = tfidf.transform(sample_news_clean)

prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("FAKE NEWS")
else:
    print("REAL NEWS")


#Streamlit

import streamlit as st




# Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon=" ")

st.title("Fake News Detection App")
st.write("Enter a news article text below and click **Predict** to check if it's Fake or Real.")

news_text = st.text_area("Paste News Text Here:", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(news_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("This News is **FAKE**")
        else:
            st.success("This News is **REAL**")


