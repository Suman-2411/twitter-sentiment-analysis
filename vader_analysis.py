import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# tried textblob first but it was not good
# from textblob import TextBlob

# cleaned up a bit 
analyzer = SentimentIntensityAnalyzer()

def analyze_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # 0.05 is from vader docs, seemed to work ok on test data
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    else:
        return 'neutral', compound

def run_vader(df):
    sentiments = []
    scores = []
    
    for tweet in df['clean_tweet']:
        sentiment, score = analyze_vader(tweet)
        sentiments.append(sentiment)
        scores.append(score)
    
    df['vader_sentiment'] = sentiments
    df['vader_score'] = scores
    
    print(df[['clean_tweet','vader_sentiment']].head(15))
    return df

#  for quick test
# df = pd.read_csv('new_train_data_s140.csv')
# run_vader(df)