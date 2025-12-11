import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn 
import nltk
import re
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean(doc):
  # doc is a string of text
  # Lets's define a regex to match special characters and digits
  regx="[^a-zA-Z\s]"
  doc=re.sub(regx," ",doc)
  # Convert to lowercase
  doc=doc.lower()
  # Tokenization
  token=nltk.word_tokenize(doc)
  # Stop word removal
  stop_words=set(stopwords.words('english'))
  filtered_tokens=[i for i in token if i not in stop_words]
  # Lemmatize
  lemmatizer=WordNetLemmatizer()
  lemmatized_tokens=[lemmatizer.lemmatize(i) for i in filtered_tokens]
  return " ".join(lemmatized_tokens)

with open("nb_model.pkl","rb") as f:
    model=pickle.load(f)

st.title("Movie Review Sentiment Analysis")

review=st.text_input("Enter Your Review:")
if st.button("Predict Sentiment"):
    review=clean(review)
    result = model.predict([review])[0]
    st.success(review)
    st.success(f"Predicted Sentiment: {result}")
