import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#To Intialize Vader moder
analyzer=SentimentIntensityAnalyzer()

def analyze_vader(text):
    #to get sentiment score and compound score
    scores=analyzer.polarity_scores(text)
    compound=scores['compound']

    #to classify sentiment
    if compound >=0.5:
        return 'positive'
    elif compound <=0.5:
        return 'negative'
    else:
        return 'neutral'
    
def run_vader(df):
    #Apply vader to every tweet
    df['vader_sentiment']=df['clean_tweet'].apply(analyze_vader)
    print(df[['clean_tweet','vader_sentiment']].head(15))
    return df

from clean_data import clean_dataset
from fetch_tweets import load_tweets

df = load_tweets('new_train_data_s140.csv')
df=clean_dataset(df)
df=run_vader(df)
df.to_csv('vader_results.csv',index=False)
print('Saved vader_results.csv!')