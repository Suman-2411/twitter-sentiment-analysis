import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df, model_name):
    if 'vader_sentiment' in df.columns:
        counts = df['vader_sentiment'].value_counts()
    else:
        counts = df['roberta_sentiment'].value_counts()

    # tried bar chart first but pie chart looked better for this
    plt.figure(figsize=(8,6))
    plt.pie(counts,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=['green','red','blue'])

    plt.title(f'{model_name} Sentiment Distribution')
    plt.savefig(f'{model_name}_distribution.png')
    plt.show()
    print(f"saved {model_name}_distribution.png")

# load the results and make charts
df_vader = pd.read_csv('vader_results.csv')
df_roberta = pd.read_csv('roberta_result.csv')

# TODO: colors dont match between both charts ,needs to fixed 

plot_sentiment_distribution(df_vader, 'VADER')
plot_sentiment_distribution(df_roberta, 'RoBERTa')
