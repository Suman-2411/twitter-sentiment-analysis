import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_roberta_model():
    # cache this or it reloads every single time, took me ages to figure out
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        max_length=512,
        truncation=True
    )

def analyze_roberta(text):
    try:
        pipe = load_roberta_model()
        result = pipe(text)[0]

        label = result['label']
        score = result['score']

        # debug - remove this later
        print(f"label: {label}, score: {score}")

        # cardiffnlp docs say LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
        if label == 'LABEL_0':
            return 'negative'
        elif label == 'LABEL_2':
            return 'positive'
        else:
            return 'neutral'

    except Exception as e:
        print(f"something broke: {e}")
        return 'neutral'