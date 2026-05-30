import streamlit as st
from transformers import pipeline
import pandas as pd

@st.cache_resource 
def load_roberta_model():
    return pipeline(
        "sentiment-analysis",
        model="cariffnlp/twitter-roberta-base-sentiment",
        max_length=512,
        truncation=True
    )

#load RoBERTa model
def analyze_roberta(text):
    try:
        sentiment_pipeline=load_roberta_model()
        result=sentiment_pipeline(text)[0]
        label=result['label']

        #to convert labels into readable format
        if label=='LABEL_0':
            return 'negative'
        elif label=='LABEL_1':
            return 'neutral'
        else:
            return 'positive'
    except:
        return 'neutral'
    
