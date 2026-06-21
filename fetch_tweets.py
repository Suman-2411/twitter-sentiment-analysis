import tweepy
import pandas as pd
from config import BEARER_TOKEN

#authenticate with twitter to fetch the tweets
client = tweepy.Client(bearer_token = BEARER_TOKEN)

#fetch the tweets based on our keyword 
def fetch_tweets(keyword, max_results = 10):

    tweets = client.search_recent_tweets(
        query=keyword,
        max_results=max_results,
        tweet_fields=['text','created_at'])
    
    #extracting the data 
    data = []
    for tweet in tweets.data:
        data.append({
            'text':tweet.text,
            'created_at': tweet.created_at
            }) 
        
    #to put tweets in table and save that table in a csv file
    df=pd.DataFrame(data)
    df.to_csv("tweets.csv")

    print(f'fetched {len(df)} tweets!')
    return df


#--------------------Kaggle Dataset--------------------------#
def load_tweets(filepath):
    #pd.read_csv is used to read the csv file
    df=pd.read_csv(filepath,
                   encoding='latin-1',
                   nrows=10000)
    
    #rename column 
    df=df.rename(columns={
        df.columns[0]: 'sentiment',
        df.columns[5]: 'text'})
    
    #keeps sentiment and text 
    df = df[['sentiment','text']]

    #convert 0=negative, 4=positive 
    df['sentiment']=df['sentiment'].map(
            {0:'negative',2:'neutral',4:'positive'})
    print(f'Loaded {len(df)} tweets!')
    print(df.head())
    return df

#test kaggle
#df=load_tweets('new_train_data_s140.csv')  