from collections import defaultdict
import json
import pandas as pd
from pymongo import MongoClient

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

def load_data(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    return data

def filter_tweets(data, db):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    users = set(user_df["userid"])
    
    filtered_users = [item for item in data if item['user_id'] in users]

    with open(f"{db}_tweets.json", "w") as f:
        json.dump(filtered_users, f, indent=4)
    
    print(len(filtered_users))

def import_tweets(db):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    users = set(user_df["userid"])

    with open(f"{db}_tweets.json", "r") as f:
        data = json.load(f)
    

    og_tweets_db = MONGO_CLIENT[db]["original_tweets"]
    retweet_in_db = MONGO_CLIENT[db]["retweets_of_in_community"]
    retweet_out_db = MONGO_CLIENT[db]["retweets_of_out_community_by_in_community"]

    for user in data:
        tweets = user['tweets']
        for tweet in tweets:
            if tweet['retweet_id']:
                # either it is retweet of in community or out community
                if tweet['retweet_user_id'] in users:
                    retweet_in_db.insert_one(tweet)
                else:
                    retweet_out_db.insert_one(tweet)
            else:
                og_tweets_db.insert_one(tweet)

    # for user in data.keys():
    #     tweets = data[user]
    #     for tweet in tweets:
    #         if tweet['retweet_id']:
    #             if tweet['retweet_user_id'] in users:
    #                 retweet_in_db.insert_one(tweet)
    #             else:
    #                 retweet_out_db.insert_one(tweet)
    #         else:
    #             og_tweets_db.insert_one(tweet)



def filter_friends(file_name, db):
    with open(file_name, "r") as file:
        data = json.load(file)
    
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    users = set(user_df["userid"])
    
    filtered_users = [item for item in data if item['user_id'] in users]

    with open(f"{db}_friends.json", "w") as f:
        json.dump(filtered_users, f, indent=4)
    

def import_followings(db):
    with open(f"{db}_friends.json", 'r') as file:
        data = json.load(file)
    
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    users = set(user_df["userid"])
    
    # for user in data:
    #     followings = [friend for friend in user["friends_ids"] if friend in users]
    #     user_collection.update_one(
    #         {"userid": user['user_id']},
    #         {"$set": {"local following list": followings}},  
    #         upsert=False
    #     )

    for user in data.keys():
        followings = [friend for friend in data[user] if friend in users]
        user_collection.update_one(
            {"userid": user},
            {"$set": {"local following list": followings}},  
            upsert=False
        )

def import_followers(db):
    with open(f"{db}_friends.json", 'r') as file:
        data = json.load(file)

    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    users = set(user_df["userid"])

    # for user in users:
    #     followers = []
    #     for follower in data:
    #         if user in follower["friends_ids"]:
    #             followers.append(follower["user_id"])
        
    #     user_collection.update_one(
    #         {"userid": user},
    #         {"$set": {"local follower list": followers}},  
    #         upsert=False
    #     )  

    for user in users:
        followers = []
        for follower in data.keys():
            if user in data[follower]:
                followers.append(follower)
        
        user_collection.update_one(
            {"userid": user},
            {"$set": {"local follower list": followers}},  
            upsert=False
        ) 
    

if __name__ == "__main__":
    communities = ["climate_change_expanded"]
    # data = load_data("Data.UserTweets.json")
    # filter_tweets(data, community)
    for community in communities:
        # import_tweets(community)
    # filter_friends("Data.Friends.json", community)
        import_followings(community)
        import_followers(community)

