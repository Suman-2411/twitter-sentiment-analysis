import pandas as pd 
import re

def clean_tweet(text):
    #remove urls 
    text=re.sub(r'https\S+','',text)
    #remove @mention
    text=re.sub(r'@\w+','',text)
    #remove special characters
    text=re.sub(r'[^a-zA-Z\s]','',text)
    #convert int lower case
    text=text.lower()
    #remove spaces
    text=text.strip()
    return text

def clean_dataset(df):
    #To apply cleaning to every tweets
    df['clean_tweet']=df['text'].apply(clean_tweet)
    #remove empty tweets
    df=df[df['clean_tweet'].str.len()>0]

    print(f'cleaned{len(df)} tweets!')
    print(df.head())
    return df

load and clean data
#from fetch_tweets import load_tweets

#df=load_tweets('new_train_data_s140.csv')
#df=clean_dataset(df)

#save the cleaned data
#df.to_csv("cleaned_tweets.csv",index=False)
#print("Saved to cleaned_tweets.csv")