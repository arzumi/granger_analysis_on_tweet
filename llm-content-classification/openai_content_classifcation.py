import os
import time
from typing import List
import ollama
from openai import OpenAI
import json
from decimal import Decimal
import pymongo
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from ollama import chat
from ollama import ChatResponse

BATCH_SIZE = 20
OPENAI_CLIENT = OpenAI(api_key="")
MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# get the orginal tweets by the core user and the tweets retweeted by the core user in the community
def get_core_user_tweets(db: str, userid=0):
    all_tweets = []
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    if userid != 0:
        core_user = userid

    original_tweet_collection = MONGO_CLIENT[db]["original_tweets"]
    original_tweet_df = pd.DataFrame(list(original_tweet_collection.find({"user_id": core_user})))
    original_tweet_df = original_tweet_df.drop_duplicates(['id'])
    # original_tweet_df.loc[:, 'id'] = original_tweet_df['id'].astype(int)
    
    retweets = MONGO_CLIENT[db]['retweets_of_in_community']
    retweets_df = pd.DataFrame(list(retweets.find({"retweet_user_id": core_user})))
    retweets_df = retweets_df.drop_duplicates(['id'])
    # retweets_df.loc[:, 'retweet_id'] = retweets_df['retweet_id'].astype(int)
    retweets_id = set(retweets_df['retweet_id'])

    for index, row in original_tweet_df.iterrows():
        # if row['id'] in retweets_id:
        #     all_tweets.append(row)
        all_tweets.append(row)

    return all_tweets

# next get all the topic for the tweets by core user
def get_all_topics(all_tweets: List, db: str):
    os.makedirs(db, exist_ok=True)
    output_file_path = f"{db}/openai/new/topic_extraction.txt"
    file = open(output_file_path, 'w')

    instruction = f"""
    Read the text below and list up to 3 topics. Each topic should contain fewer than 3 words. Ensure you only return the topic and nothing more.
    The desired output format:
    Topic 1: xxx\nTopic 2: xxx\nTopic 3: xxx
    """

    unique_topics = set()
    count = 0

    NUM_TWEETS = len(all_tweets)
    batch_iter = 0
    num_iters = (NUM_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE

    while batch_iter < num_iters and batch_iter * BATCH_SIZE <= NUM_TWEETS:
        text_input = ""
        for i in range(batch_iter * BATCH_SIZE, min((batch_iter + 1) * BATCH_SIZE, len(all_tweets))):
            text_input += f"Tweet {i + 1}: {all_tweets[i]['text']}\n"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text_input}
        ]

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.5
        )

        topics = response.choices[0].message.content.strip().split('\n')

        for topic in topics:
            topic_text = topic.split(": ")[1]
            unique_topics.add(topic_text)

        batch_iter += 1
    
    for topic in unique_topics:
        file.write(f"{topic}\n")
    file.close()

# from all the topics group them into similar topics
def get_summarized_topics(db: str):
    file_path = f"{db}/openai/new/topic_extraction.txt"
    with open(file_path, 'r') as file:
        topics = file.read()
    
    output_file = f"{db}/openai/new/summary.txt"
    file = open(output_file, 'w')

    instruction = f"""
    Group the following topics into 15 to 25 meaningful clusters based on semantic similarity. 
    Ensure the groups are detailed but not overly broad. If topics are distinct, do not merge them into larger categories. 
    Keep groupings specific but not overly detailed. Avoid merging unrelated topics.
    Provide a clear and descriptive label for each group in a single line, with no numbering or additional explanation.
    """

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": topics}
    ]

    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        temperature=0.5
    )

    topics = response.choices[0].message.content.strip()
    for topic in topics:
        file.write(topic)
    file.close()

# then classify all the OG tweets and all the retweets in community & out community
def get_topic_per_tweet_og(db: str):
    # first get all tweets
    all_tweets = get_all_tweets(db)

    # get all the summarized topics
    summary_file = f"{db}/openai/new/summary.txt"
    with open(summary_file, 'r') as file:
        summarized_topics = file.read()

    instruction = f"""
    Classify the following input text into one of the following categories:\n
    {summarized_topics}\n\n
    The ENTIRE string before the colon is the id of the tweet and the text inside the square brackets is the text for the tweet. \n
    If the tweet cannot be classified then return the topic as None. Follow the format for tweets with no topic strictly:
    id: None \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    Be sure to state only the topic without the number. Only return the topic and the id.
    Please follow the following format strictly:
    id: Topic \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    ALSO PLEASE RETURN THE ENTIRE ID NOT JUST THE LAST CHUNK! The ID is quite long so please return EVERYTHING!!! PLEASE DO THIS!!!! 
    \n Please dont be stupid and return the word id without replacing it with the ID that I provided. \n
    PLEASE BUT PLEASE RETURN THE ENTIRE ID WHICH IS THE ENTIRE STRING BEFORE THE COLON!!!!!!!!! PLEASE BE SMART!!!!!!
    Also try your best to classify the text into one of the categories that are provided!!!!
    Lastly, PLEASE RETURN ALL THE TWEETS THAT I GAVE YOU DONT MISS ONE!!! RETURN ALL 20!!!
    """

    #then classify them one by one and create a json
    # ! only get the tweets we are interested in
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    
    core = user_collection.find_one({"rank": 0})
    core_user = core["userid"]

    consumers = [x for x in core["local follower list"]]

    producers = core["local following list"]
    producer_df = user_df[user_df['userid'].isin(producers)]
    producer_df = producer_df[producer_df['local following list'].apply(lambda x: core_user in x)]

    producers = producer_df['userid'].to_list()
    # ! only interested tweets

    unfiltered_tweets = all_tweets['og']
    tweets = unfiltered_tweets[unfiltered_tweets['user_id'].isin(producers + [core_user])]
    output_file = f"{db}/openai/new/og_tweets_classification.json"

    NUM_TWEETS = len(tweets)
    batch_iter = 0
    num_iters = (NUM_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE
    classifications = {}

    total_tweets = 0

    with open(output_file, 'a') as file:
        while batch_iter <= num_iters:
            text_input = ""
            tweet_ids = set()
            for i in range(batch_iter * BATCH_SIZE, min((batch_iter + 1) * BATCH_SIZE, NUM_TWEETS)):
                id_value = tweets.iloc[i]['id']
                # tweet_id = int(Decimal(id_value))
                tweet_id = id_value
                tweet_ids.add(tweet_id)
                tweet_text = tweets.iloc[i]['text']
                text_input += f"{tweet_id}: [{tweet_text}]\n"
            
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text_input}
            ]

            try:
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.5
                )
            except Exception as e:
                print(f"Error during API call: {e}")
                batch_iter += 1
                time.sleep(10)
                continue

            topics = response.choices[0].message.content.strip().split('\n')
            batch_classifications = {}
            for topic in topics:
                try:
                    tweet_id, topic_text = topic.split(": ", 1)
                    # batch_classifications[int(tweet_id.strip())] = topic_text.strip()
                    batch_classifications[tweet_id.strip()] = topic_text.strip()
                except ValueError:
                    print(f"Unexpected format in response: {topic}")
                    print(f"Response content: {response.choices[0].message.content}")

            classifications.update(batch_classifications)
            json.dump(batch_classifications, file, indent=4)
            file.write("\n")

            total_tweets += len(batch_classifications)
            print(f"Processed batch {batch_iter + 1}/{num_iters}, Total processed tweets: {total_tweets}")

            batch_iter += 1
            time.sleep(10)

    print(f"Total tweets classified: {total_tweets} / {len(tweets)}")

def get_topic_per_tweet_in(db: str):
    # first get all the tweets
    all_tweets = get_all_tweets(db)
    all_tweets = all_tweets['retweet_in']
    # all_tweets['retweet_id'] = all_tweets['retweet_id'].astype('int64')
    # all_tweets['id'] = all_tweets['id'].astype('int64')

    with open(f"{db}/openai/new/{db}_word_og.json") as file:
        og_tweets_classification = json.load(file)

    classification = {}
    not_found_ids = []
    for index, tweet in all_tweets.iterrows():
        tweet_id = str(tweet['retweet_id'])
        if tweet_id in og_tweets_classification:
            classification[tweet['id']] = og_tweets_classification[tweet_id]
        else:
            not_found_ids.append(tweet['id'])

    with open(f"{db}/openai/new/{db}_word_in.json", 'w') as file:
        json.dump(classification, file, indent=4)
    
    with open(f"{db}/openai/new/not_found_retweet_in.txt", 'w') as file2:
        for id in not_found_ids:
            file2.write(str(id) + '\n')

def get_topic_per_tweet_out(db: str):
    # first get all tweets
    all_tweets = get_all_tweets(db)

    # get all the summarized topics
    summary_file = f"{db}/openai/new/summary.txt"
    with open(summary_file, 'r') as file:
        summarized_topics = file.read()

    instruction = f"""
    Classify the following input text into one of the following categories:\n
    {summarized_topics}\n\n
    The ENTIRE string before the colon is the id of the tweet and the text inside the square brackets is the text for the tweet. \n
    If the tweet cannot be classified then return the topic as None. Follow the format for tweets with no topic strictly:
    id: None \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    Be sure to state only the topic without the number. Only return the topic and the id.
    Please follow the following format strictly:
    id: Topic \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    ALSO PLEASE RETURN THE ENTIRE ID NOT JUST THE LAST CHUNK! The ID is quite long so please return EVERYTHING!!! PLEASE DO THIS!!!! 
    \n Please dont be stupid and return the word id without replacing it with the ID that I provided. \n
    PLEASE BUT PLEASE RETURN THE ENTIRE ID WHICH IS THE ENTIRE STRING BEFORE THE COLON!!!!!!!!! PLEASE BE SMART!!!!!!
    Also try your best to classify the text into one of the categories that are provided!!!!
    Lastly, PLEASE RETURN ALL THE TWEETS THAT I GAVE YOU DONT MISS ONE!!! RETURN ALL 20!!!
    """

    #then classify them one by one and create a json
    # ! only get the tweets we are interested in
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    
    core = user_collection.find_one({"rank": 0})
    core_user = core["userid"]

    consumers = [x for x in core["local follower list"]]
    # ! only interested tweets

    unfiltered_tweets = all_tweets['retweet_in']
    tweets = unfiltered_tweets[unfiltered_tweets['user_id'].isin(consumers + [core_user])]

    output_file = f"{db}/openai/new/retweet_in_tweets_classification.json"

    NUM_TWEETS = len(tweets)
    batch_iter = 0
    num_iters = (NUM_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE
    classifications = {}

    total_tweets = 0

    with open(output_file, 'a') as file:
        while batch_iter <= num_iters:
            text_input = ""
            tweet_ids = set()
            for i in range(batch_iter * BATCH_SIZE, min((batch_iter + 1) * BATCH_SIZE, NUM_TWEETS)):
                id_value = tweets.iloc[i]['id']
                # tweet_id = int(Decimal(id_value))
                tweet_id = id_value
                tweet_ids.add(tweet_id)
                tweet_text = tweets.iloc[i]['text']
                text_input += f"{tweet_id}: [{tweet_text}]\n"
            
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text_input}
            ]

            try:
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.5
                )
            except Exception as e:
                print(f"Error during API call: {e}")
                batch_iter += 1
                time.sleep(10)
                continue

            topics = response.choices[0].message.content.strip().split('\n')
            batch_classifications = {}
            for topic in topics:
                try:
                    tweet_id, topic_text = topic.split(": ", 1)
                    # batch_classifications[int(tweet_id.strip())] = topic_text.strip()
                    batch_classifications[tweet_id.strip()] = topic_text.strip()
                except ValueError:
                    print(f"Unexpected format in response: {topic}")
                    print(f"Response content: {response.choices[0].message.content}")
            
            classifications.update(batch_classifications)
            json.dump(batch_classifications, file, indent=4)
            file.write("\n")

            total_tweets += len(batch_classifications)
            print(f"Processed batch {batch_iter + 1}/{num_iters}, Total processed tweets: {total_tweets}")

            batch_iter += 1           
            time.sleep(5)

    print(f"Total tweets classified: {total_tweets} / {len(tweets)}")

# helper
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

def classify_ignored_tweets(file_path, db, types):
    all_tweets = get_all_tweets(db)
    all_tweets = all_tweets[types]
    # all_tweets['id'] = all_tweets['id'].astype('int64')
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    ids_set = list(data.keys())
    filtered_tweets = all_tweets[~all_tweets['id'].isin(ids_set)]

    summary_file = f"{db}/openai/new/summary.txt"
    with open(summary_file, 'r') as file:
        summarized_topics = file.read()

    output_file = f"{db}/openai/new/{types}_tweets_classification_ignored.json"

    instruction = f"""
    Classify the following input text into one of the following categories:\n
    {summarized_topics}\n\n
    The ENTIRE string before the colon is the id of the tweet and the text inside the square brackets is the text for the tweet. \n
    If the tweet cannot be classified then return the topic as None. Follow the format for tweets with no topic strictly:
    id: None \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    Be sure to state only the topic without the number. Only return the topic and the id.
    Please follow the following format strictly:
    id: Topic \n
    Here id is the id that I provided for the tweets, so replace the word id with the id that i provided. PLEASE DO THIS!!!!
    ALSO PLEASE RETURN THE ENTIRE ID NOT JUST THE LAST CHUNK! The ID is quite long so please return EVERYTHING!!! PLEASE DO THIS!!!! 
    \n Please dont be stupid and return the word id without replacing it with the ID that I provided. \n
    PLEASE BUT PLEASE RETURN THE ENTIRE ID WHICH IS THE ENTIRE STRING BEFORE THE COLON!!!!!!!!! PLEASE BE SMART!!!!!!
    Also try your best to classify the text into one of the categories that are provided!!!!
    Lastly, PLEASE RETURN ALL THE TWEETS THAT I GAVE YOU DONT MISS ONE!!! RETURN ALL 20!!!
    """

    NUM_TWEETS = len(filtered_tweets)
    batch_iter = 0
    num_iters = (NUM_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE
    classifications = {}

    total_tweets = 0

    with open(output_file, 'a') as file:
        while batch_iter <= num_iters:
            text_input = ""
            tweet_ids = set()
            for i in range(batch_iter * BATCH_SIZE, min((batch_iter + 1) * BATCH_SIZE, NUM_TWEETS)):
                id_value = filtered_tweets.iloc[i]['id']
                # tweet_id = int(Decimal(id_value))
                tweet_id = id_value
                tweet_ids.add(tweet_id)
                tweet_text = filtered_tweets.iloc[i]['text']
                text_input += f"{tweet_id}: [{tweet_text}]\n"
            
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text_input}
            ]

            try:
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.5
                )
            except Exception as e:
                print(f"Error during API call: {e}")
                batch_iter += 1
                time.sleep(10)
                continue

            topics = response.choices[0].message.content.strip().split('\n')
            batch_classifications = {}
            for topic in topics:
                try:
                    tweet_id, topic_text = topic.split(": ", 1)
                    # batch_classifications[int(tweet_id.strip())] = topic_text.strip()
                    batch_classifications[tweet_id.strip()] = topic_text.strip()
                except ValueError:
                    print(f"Unexpected format in response: {topic}")
                    print(f"Response content: {response.choices[0].message.content}")

            classifications.update(batch_classifications)
            json.dump(batch_classifications, file, indent=4)
            file.write("\n")

            ignored_file = f"{db}/openai/new/{types}_ignored.txt"
            with open(ignored_file, 'a') as file2:  
                for id in tweet_ids:
                    if id not in batch_classifications:
                        file2.write(f"{id}\n")

            total_tweets += len(batch_classifications)
            print(f"Processed batch {batch_iter + 1}/{num_iters}, Total processed tweets: {total_tweets}")

            batch_iter += 1
            time.sleep(10)

    print(f"Total tweets classified: {total_tweets}")
    

def flatten_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)  # Load the JSON file

    # flatten the list of dictionaries
    flattened_data = {}
    for dictionary in data:
        flattened_data.update(dictionary)  # merge each dictionary into the result

    # write the flattened dictionary back to the file
    with open(f"{file_path[:-5]}_new.json", "w") as file:
        json.dump(flattened_data, file, indent=4)




if __name__ == "__main__":
    communities = ["science_fiction_expanded", "game_development", "urban_planning", "neuroscience_expanded"]
    for community in communities:
        # all_tweets = get_core_user_tweets(community)
        # print(len(all_tweets))
        # get_all_topics(all_tweets, community)
        # get_summarized_topics(community)
        # get_topic_per_tweet_og(community)
        get_topic_per_tweet_in(community)
        # get_topic_per_tweet_out(community)
    # get_all_tweets(community)
    # classify_ignored_tweets(f"{community}/openai/new/retweet_out_tweets_classification_new.json", community, "retweet_out")
    # classify_ignored_tweets(f"{community}/openai/new/og_tweets_classification_new.json", community, "og")
        # flatten_json(f"{community}/openai/new/retweet_in_tweets_classification.json")
        # flatten_json(f"{community}/openai/new/og_tweets_classification.json")
