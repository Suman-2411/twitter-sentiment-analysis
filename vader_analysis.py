import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# tried textblob first but the scores were all over the place
# from textblob import TextBlob
# cleaned up a bit 
analyzer = SentimentIntensityAnalyzer()

def analyze_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # 0.05 threshold from vader docs, seemed to work ok on test data
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    else:
        return 'neutral', compound

def run_vader(df):
    # TODO: maybe add progress bar here for large datasets
    df['vader_sentiment'] = df['clean_tweet'].apply(analyze_vader)
    print(df[['clean_tweet', 'vader_sentiment']].head(15))
    return df

# quick test
# df = pd.read_csv('new_train_data_s140.csv')
# run_vader(df)