import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def calculate_metrics(df, model_name):
    # Get true labels
    true_labels = df['sentiment']
    
    # Get predicted labels
    if model_name == 'VADER':
        pred_labels = df['vader_sentiment']
    else:
        pred_labels = df['roberta_sentiment']
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, 
                  average='weighted',
                  zero_division=0)
    precision = precision_score(true_labels, pred_labels,
                               average='weighted',
                               zero_division=0)
    recall = recall_score(true_labels, pred_labels,
                         average='weighted',
                         zero_division=0)
    
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }