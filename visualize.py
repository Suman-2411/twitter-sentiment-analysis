import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df,model_name):

    #To count sentiment
    if 'vader_sentiment' in df.columns:
        counts=df['vader_sentiment'].value_counts()
    else:
        counts=df['roberta_sentiment'].value_counts()

    #To create pie chart
    plt.figure(figsize=(8,6))
    plt.pie(counts,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=['green','red','blue'])
    plt.title(f'{model_name} Sentiment Distribution')
    plt.savefig(f'{model_name}_distribution.png')
    plt.show()
    print(f"Saver{model_name}_distribution.png!")

#To load and visualize
df_vader=pd.read_csv('vader_results.csv')
df_roberta=pd.read_csv('roberta_result.csv')

#plot VADER distribution
plot_sentiment_distribution(df_vader, 'VADER')

#plot RoBERTa distribution
plot_sentiment_distribution(df_roberta, 'RoBERTa')

