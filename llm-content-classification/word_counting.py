import json
import re
from collections import defaultdict
from pymongo import MongoClient
import pandas as pd

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# example of how topics should look like
topics = {
    "American Democracy and Political Structure": [
        "democracy", "uspolitics", "government"
    ]
}

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def get_all_tweets(db: str):
    all_tweets = {}
    original_tweet_collection = MONGO_CLIENT[db]["original_tweets"]
    original_tweet_df = pd.DataFrame(list(original_tweet_collection.find()))
    original_tweet_df = original_tweet_df.drop_duplicates(['id'])

    retweet_in_collection = MONGO_CLIENT[db]["retweets_of_in_community"]
    retweet_in_df = pd.DataFrame(list(retweet_in_collection.find()))
    retweet_in_df = retweet_in_df.drop_duplicates(['id'])
    # retweet_in_df.loc[:, 'id'] = retweet_in_df['id'].astype(int)

    retweet_out_collection = MONGO_CLIENT[db]["retweets_of_out_community_by_in_community"]
    retweet_out_df = pd.DataFrame(list(retweet_out_collection.find()))
    retweet_out_df = retweet_out_df.drop_duplicates(['id'])
    # retweet_out_df.loc[:, 'id'] = retweet_out_df['id'].astype(int)

    all_tweets["og"] = original_tweet_df
    all_tweets["retweet_in"] = retweet_in_df
    all_tweets["retweet_out"] = retweet_out_df

    return all_tweets

def classifyi_tweet_og(db):
    tweet_topic_mapping = {}

    all_tweets = get_all_tweets(db)
    original_tweets = all_tweets['og']

    for index, tweet in original_tweets.iterrows():
        text = tweet["text"]
        tweet_id = tweet["id"]
        
        tokens = tokenize(text)
        
        topic_counts = {}
        
        for topic, keywords in topics.items():
            count = sum(tokens.count(keyword.lower()) for keyword in keywords)
            topic_counts[topic] = count

        best_topic = max(topic_counts, key=topic_counts.get)
        if topic_counts[best_topic] > 0:
            tweet_topic_mapping[tweet_id] = best_topic
    
    with open(f"{db}_word_og.json", "w") as file:
        json.dump(tweet_topic_mapping, file)

def classifyi_tweet_out(db):
    tweet_topic_mapping = {}

    all_tweets = get_all_tweets(db)
    original_tweets = all_tweets['retweet_out']

    for index, tweet in original_tweets.iterrows():
        text = tweet["text"]
        tweet_id = tweet["id"]
        
        tokens = tokenize(text)
        
        topic_counts = {}
        
        for topic, keywords in topics.items():
            count = sum(tokens.count(keyword.lower()) for keyword in keywords)
            topic_counts[topic] = count

        # (if none of the topics has at least one match, we ignore the tweet)
        best_topic = max(topic_counts, key=topic_counts.get)
        if topic_counts[best_topic] > 0:
            tweet_topic_mapping[tweet_id] = best_topic
    
    with open(f"{db}_word_out.json", "w") as file:
        json.dump(tweet_topic_mapping, file)

if __name__ == "__main__":
    community = "neuroscience_expanded"
    classifyi_tweet_og(community)
    classifyi_tweet_out(community)