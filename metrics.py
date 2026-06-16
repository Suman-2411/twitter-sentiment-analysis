import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# accuracy alone was not enough
# so added f1 score too

def calculate_metrics(df, model_name):
    true_labels = df['sentiment']
    
    # get the right column
    if model_name == 'VADER':
        pred_labels = df['vader_sentiment']
    else:
        pred_labels = df['roberta_sentiment']
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # weighted works better for unequal classes
    f1 = f1_score(true_labels, pred_labels,
                  average='weighted',
                  zero_division=0)
    
    precision = precision_score(true_labels, pred_labels,
                               average='weighted',
                               zero_division=0)
    
    recall = recall_score(true_labels, pred_labels,
                         average='weighted',
                         zero_division=0)
    
    # TODO: add confusion matrix later
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }