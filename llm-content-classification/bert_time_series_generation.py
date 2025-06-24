# first clean up the classifcations 
import json
import pandas as pd
from pymongo import MongoClient
MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

def remove_none_type(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    new_classication = {}
    for key, value in data.items():
        if value:
            new_classication[key] = value
    
    with open(f"{file_name[:-5]}_clean.json", 'w') as file:
        json.dump(new_classication, file, indent=4)


# create time series for each user
def generate_time_series_supply():
    # first load all the topic that we generated
    with open("ml_community_1/bert/new/reduced_topics_og.txt", 'r') as file:
        topics = [topic.strip() for topic in file.read().splitlines()]
    
    with open("ml_community_1/bert/new/og_tweets_bert_classification_clean.json") as file:
        tweets_mapping = json.load(file)
    
    # get all the user information and the tweets that they posted 
    db = MONGO_CLIENT["ml_community_1"]
    user_collection = db["user_info"]
    user_df = pd.DataFrame(list(user_collection.find()))
    user_df['userid'] = user_df['userid'].astype(int)

    original_tweet = db["original_tweets"]
    original_df = pd.DataFrame(list(original_tweet.find()))
    original_df['user_id'] = original_df['user_id'].astype(int)
    original_df['created_at'] = pd.to_datetime(original_df['created_at']).dt.date
    original_df['id'] = original_df['id'].astype(int)
    original_df = original_df.drop_duplicates(['id'])

    og_tweet_time_series = db["og_tweet_time_series_bert"]

    # from all the original tweets find the oldest tweets and the newest tweet
    min_date = original_df['created_at'].min()
    max_date = original_df['created_at'].max()

    # this is the range that we will create the time series
    time_range = pd.date_range(start=min_date, end=max_date)
    
    # iterate over each user and crate time series per user
    for user in user_df['userid'].unique():
        # filter the user tweets based on the user we are looking at rn
        user_tweets = original_df[original_df['user_id'] == user]
        user_time_series_df = pd.DataFrame(index=time_range, columns=topics)
        user_time_series_df = user_time_series_df.fillna(0)

        for _, tweet in user_tweets.iterrows():
            tweet_date = tweet['created_at']
            tweet_id_str = str(tweet['id'])
            if tweet_id_str in tweets_mapping:
                topic = tweets_mapping[tweet_id_str]
                topic = topic.strip()
                if topic in user_time_series_df.columns:
                    user_time_series_df.at[pd.Timestamp(tweet_date), topic] += 1
        
        user_time_series_df = user_time_series_df.reset_index()
        user_time_series_df.rename(columns={'index': 'date'}, inplace=True)

        time_series_records = []
        for _, row in user_time_series_df.iterrows():
            date = row['date']
            topics_counts = {topic: int(row[topic]) for topic in topics}
           
            time_series_records.append({
                'date': date,
                'topics': topics_counts
            })
        
        user_id = int(user)

        user_document = {
            'user_id': user_id,
            'time_series': time_series_records
        }

        og_tweet_time_series.insert_one(user_document)

# create time series for each user for retweet_in
def generate_time_series_demand_in():
    # first load all the topic that we generated
    with open("ml_community_1/bert/new/reduced_topics_og.txt", 'r') as file:
        topics = [topic.strip() for topic in file.read().splitlines()]
    
    with open("ml_community_1/bert/new/retweet_in_tweets_bert_classification_clean.json") as file:
        tweets_mapping = json.load(file)
    
    # get all the user information and the tweets that they posted 
    db = MONGO_CLIENT["ml_community_1"]
    user_collection = db["user_info"]
    user_df = pd.DataFrame(list(user_collection.find()))
    user_df['userid'] = user_df['userid'].astype(int)

    retweet_in = db["retweets_of_in_community"]
    retweet_in_df = pd.DataFrame(list(retweet_in.find()))
    retweet_in_df['user_id'] = retweet_in_df['user_id'].astype(int)
    retweet_in_df['created_at'] = pd.to_datetime(retweet_in_df['created_at']).dt.date
    retweet_in_df['id'] = retweet_in_df['id'].astype(int)
    retweet_in_df = retweet_in_df.drop_duplicates(['id'])

    retweet_tweet_time_series = db["retweet_in_time_series_bert"]

    # from all the original tweets find the oldest tweets and the newest tweet
    min_date = retweet_in_df['created_at'].min()
    max_date = retweet_in_df['created_at'].max()

    # this is the range that we will create the time series
    time_range = pd.date_range(start=min_date, end=max_date)
    
    # iterate over each user and crate time series per user
    for user in user_df['userid'].unique():
        # filter the user tweets based on the user we are looking at rn
        user_tweets = retweet_in_df[retweet_in_df['user_id'] == user]
        user_time_series_df = pd.DataFrame(index=time_range, columns=topics)
        user_time_series_df = user_time_series_df.fillna(0)

        for _, tweet in user_tweets.iterrows():
            tweet_date = tweet['created_at']
            tweet_id_str = str(tweet['id'])
            if tweet_id_str in tweets_mapping:
                topic = tweets_mapping[tweet_id_str]
                topic = topic.strip()
                if topic in user_time_series_df.columns:
                    user_time_series_df.at[pd.Timestamp(tweet_date), topic] += 1
        
        user_time_series_df = user_time_series_df.reset_index()
        user_time_series_df.rename(columns={'index': 'date'}, inplace=True)

        time_series_records = []
        for _, row in user_time_series_df.iterrows():
            date = row['date']
            topics_counts = {topic: int(row[topic]) for topic in topics}
           
            time_series_records.append({
                'date': date,
                'topics': topics_counts
            })
        
        user_id = int(user)

        user_document = {
            'user_id': user_id,
            'time_series': time_series_records
        }

        retweet_tweet_time_series.insert_one(user_document)

# create time series for each user for retweet_out
def generate_time_series_demand_out():
    # first load all the topic that we generated
    with open("ml_community_1/bert/new/reduced_topics_og.txt", 'r') as file:
        topics = [topic.strip() for topic in file.read().splitlines()]
    
    with open("ml_community_1/bert/new/retweet_out_tweets_bert_classification_clean.json") as file:
        tweets_mapping = json.load(file)
    
    # get all the user information and the tweets that they posted 
    db = MONGO_CLIENT["ml_community_1"]
    user_collection = db["user_info"]
    user_df = pd.DataFrame(list(user_collection.find()))
    user_df['userid'] = user_df['userid'].astype(int)

    retweet_out = db["retweets_of_out_community_by_in_community"]
    retweet_out_df = pd.DataFrame(list(retweet_out.find()))
    retweet_out_df['user_id'] = retweet_out_df['user_id'].astype(int)
    retweet_out_df['created_at'] = pd.to_datetime(retweet_out_df['created_at']).dt.date
    retweet_out_df['id'] = retweet_out_df['id'].astype(int)
    retweet_out_df = retweet_out_df.drop_duplicates(['id'])

    retweet_tweet_time_series = db["retweet_out_time_series_bert"]

    # from all the original tweets find the oldest tweets and the newest tweet
    min_date = retweet_out_df['created_at'].min()
    max_date = retweet_out_df['created_at'].max()

    # this is the range that we will create the time series
    time_range = pd.date_range(start=min_date, end=max_date)
    
    # iterate over each user and crate time series per user
    for user in user_df['userid'].unique():
        # filter the user tweets based on the user we are looking at rn
        user_tweets = retweet_out_df[retweet_out_df['user_id'] == user]
        user_time_series_df = pd.DataFrame(index=time_range, columns=topics)
        user_time_series_df = user_time_series_df.fillna(0)

        for _, tweet in user_tweets.iterrows():
            tweet_date = tweet['created_at']
            tweet_id_str = str(tweet['id'])
            if tweet_id_str in tweets_mapping:
                topic = tweets_mapping[tweet_id_str]
                topic = topic.strip()
                if topic in user_time_series_df.columns:
                    user_time_series_df.at[pd.Timestamp(tweet_date), topic] += 1
        
        user_time_series_df = user_time_series_df.reset_index()
        user_time_series_df.rename(columns={'index': 'date'}, inplace=True)

        time_series_records = []
        for _, row in user_time_series_df.iterrows():
            date = row['date']
            topics_counts = {topic: int(row[topic]) for topic in topics}
           
            time_series_records.append({
                'date': date,
                'topics': topics_counts
            })
        
        user_id = int(user)

        user_document = {
            'user_id': user_id,
            'time_series': time_series_records
        }

        retweet_tweet_time_series.insert_one(user_document)


if __name__ == "__main__":
    # remove_none_type("ml_community_1/bert/new/retweet_out_tweets_bert_classification.json")
    #get_summary_topics()
    # generate_time_series_supply()
    # generate_time_series_demand_in()
    generate_time_series_demand_out()
