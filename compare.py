import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_models(df_vader,df_roberta):

    #To count sentiment
    vader_counts=df_vader['vader_sentiment'].value_counts()
    roberta_counts=df_roberta['roberta_sentiment'].value_counts()

    #Categories
    categories=['negative','positive','neutral']

    #values
    vader_vals=[vader_counts.get(c,0) for c in categories]
    roberta_vals=[roberta_counts.get(c,0) for c in categories]

    #To create bar chart
    x=np.arange(len(categories))
    width=0.35

    fig, ax=plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, vader_vals, width,
           label='VADER', color ='blue', alpha=0.7)
    ax.bar(x + width/2,roberta_vals, width,
           label='RoBERTa', color = 'red', alpha=0.7)
    
    #Labels
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('VADER vs RoBERTa Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.savefig('model_comparison.png')
    plt.show()
    print("Savedmodel_comparison")

#Load and compare the results
df_vader=pd.read_csv('vader_results.csv')
df_roberta=pd.read_csv('roberta_result.csv')
compare_models(df_vader,df_roberta)
