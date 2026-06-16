import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_roberta_model():
    # cache - without this it reload every time and that takes too much time and very slow
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

        # LABEL_0 is negative, LABEL_1 is neutral, LABEL_2 is positive
        if label == 'LABEL_0':
            return 'negative'
        elif label == 'LABEL_2':
            return 'positive'
        else:
            return 'neutral'

    except Exception as e:
        #if something goes wrong just return neutral
        print(f"something broke: {e}")
        return 'neutral'