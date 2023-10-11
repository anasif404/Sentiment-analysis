import pandas as pd
import tweepy
import configparser

from tabulate import tabulate

#CONFIG
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#AUTHENTICATION
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#SEARCH TWEETS

keywords = 'keys'
limit = 200
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count = 100, tweet_mode='extended').items(limit)

columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

print(tabulate(df, headers='keys', tablefmt='psql'))