from decimal import Decimal
import json
import re
import string
from typing import List
import pymongo
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
nltk.download('stopwords')
nltk.download('wordnet')

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# get the orginal tweets by the core user and the tweets retweeted by the core user in the community
def get_core_user_tweets(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    original_tweet_collection = MONGO_CLIENT[db]["original_tweets"]
    original_tweet_df = pd.DataFrame(list(original_tweet_collection.find({"user_id": core_user})))
    original_tweet_df = original_tweet_df.drop_duplicates(['id'])
    original_tweet_df.loc[:, 'id'] = original_tweet_df['id'].astype(int)
    
    retweets = MONGO_CLIENT[db]['retweets_of_in_community']
    retweets_df = pd.DataFrame(list(retweets.find({"retweet_user_id": core_user})))
    retweets_df = retweets_df.drop_duplicates(['id'])
    retweets_df.loc[:, 'retweet_id'] = retweets_df['retweet_id'].astype(int)
    retweets_id = set(retweets_df['retweet_id'])

    filtered_tweets = original_tweet_df[original_tweet_df['id'].isin(retweets_id)]
    return filtered_tweets

# clean up the tweets
def get_clean_tweets(all_tweets: pd.DataFrame):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    all_tweets['clean_text'] = all_tweets['text'].str.lower().str.strip()
    
    # 2. Remove "RT" at the start
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(r'^rt\s+', '', regex=True)
    
    # 3. Remove mentions
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(r'@\S+', '', regex=True)
    
    # 4. Remove URLs
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(r'http\S+|www\S+', ' ', regex=True)
    
    # 5. Remove punctuation
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(f"[{re.escape(string.punctuation)}]", ' ', regex=True)
    
    # 6. Remove non-alphabetic characters
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    # 7. Remove extra whitespace
    all_tweets['clean_text'] = all_tweets['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 8. Remove stopwords and lemmatize
    all_tweets['tokens'] = all_tweets['clean_text'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]
    )

    all_tweets["processed_text"] = all_tweets['tokens'].apply(lambda tokens: ' '.join(tokens))
    return all_tweets


# then train the model and get the topic
def get_topics(all_tweets: pd.DataFrame, db: str):
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    umap_model = UMAP(n_neighbors=5, n_components=2, metric="cosine")

    bert_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=3,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )

    all_tweets_final = all_tweets["processed_text"].tolist()
    topics, probabilities = bert_model.fit_transform(all_tweets_final)

    fig = bert_model.visualize_topics()
    fig.write_html(f"{db}/bert/new/topics_visualization_tweets.html")

    topic_info = bert_model.get_topic_info()
    print(topic_info)
    all_topics = []
    for topic_id in topic_info['Topic']:
        if topic_id != -1:  
            topic_words = bert_model.get_topic(topic_id)  
            formatted_topic = " ".join([word for word, _ in topic_words])
            all_topics.append(formatted_topic)

    output_path = f"{db}/bert/new/topics_tweets.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_topics))

    save_path = f"{db}/bert/new/bertopic_model/model.pkl"
    bert_model.save(save_path)

# reduce # of topics
def reduce_topics(cleaned_tweets, db):
    model = BERTopic.load(f"{db}/bert/new/bertopic_model/model.pkl")

    cleaned_tweets_final = cleaned_tweets["processed_text"].tolist()
    model.reduce_topics(cleaned_tweets_final, nr_topics=8)
    fig = model.visualize_topics()
    fig.write_html(f"{db}/bert/new/topics_visualization_reduced_og.html")

    topics = model.get_topics()
    with open(f"{db}/bert/new/reduced_topics_og.txt", "w") as f:
        for topic_num, topic_words in topics.items():
            topic_line = " ".join(word for word, _ in topic_words)
            f.write(f"{topic_line}\n")
    
    save_path = f"{db}/bert/new/bertopic_model/new_model.pkl"
    model.save(save_path)

# classify the tweets
def get_topic_per_tweet_og(db, batch_size=100):
    all_tweets = get_all_tweets("ml_community_1")
    all_tweets = all_tweets["og"]
    tweets = get_clean_tweets(all_tweets)

    classifications = {}
    output_file = f"{db}/bert/new/og_tweets_bert_classification.json"
    file = open(output_file, 'w')

    texts = tweets["processed_text"].tolist()
    ids = tweets['id'].apply(lambda id_value: int(Decimal(id_value))).tolist()

    model = BERTopic.load(f"{db}/bert/new/bertopic_model/new_model.pkl")

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        
        batch_topic_ids = model.transform(batch_texts)[0]

        for i, topic_id in enumerate(batch_topic_ids):
            tweet_id = batch_ids[i]
            topic_id = topic_id
            if topic_id != -1:
                topic_words = model.get_topic(topic_id)
                topic_words_str = ' '.join([word for word, _ in topic_words])
                classifications[tweet_id] = topic_words_str
            else:
                classifications[tweet_id] = None
    json.dump(classifications, file, indent=4)
    file.close()

def get_topic_per_tweet_in(db):
    all_tweets = get_all_tweets(db)
    all_tweets = all_tweets['retweet_in']
    all_tweets['retweet_id'] = all_tweets['retweet_id'].astype('int64')
    all_tweets['id'] = all_tweets['id'].astype('int64')

    with open(f"{db}/bert/new/og_tweets_bert_classification.json") as file:
        og_tweets_classification = json.load(file)

    
    classification = {}
    not_found_ids = []
    for index, tweet in all_tweets.iterrows():
        tweet_id = str(tweet['retweet_id'])
        if tweet_id in og_tweets_classification:
            classification[tweet['id']] = og_tweets_classification[tweet_id]
        else:
            not_found_ids.append(tweet['id'])

    with open(f"{db}/bert/new/retweet_in_tweets_bert_classification.json", 'w') as file:
        json.dump(classification, file, indent=4)
    
    with open(f"{db}/bert/new/not_found_bert_retweet_in.txt", 'w') as file2:
        for id in not_found_ids:
            file2.write(str(id) + '\n')

def get_topic_per_tweet_out(db, batch_size=100):
    all_tweets = get_all_tweets("ml_community_1")
    all_tweets = all_tweets["retweet_out"]
    tweets = get_clean_tweets(all_tweets)

    classifications = {}
    output_file = f"{db}/bert/new/retweet_out_tweets_bert_classification.json"
    file = open(output_file, 'w')

    texts = tweets["processed_text"].tolist()
    ids = tweets['id'].apply(lambda id_value: int(Decimal(id_value))).tolist()

    model = BERTopic.load(f"{db}/bert/new/bertopic_model/new_model.pkl")

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        
        batch_topic_ids = model.transform(batch_texts)[0]

        for i, topic_id in enumerate(batch_topic_ids):
            tweet_id = batch_ids[i]
            topic_id = topic_id
            if topic_id != -1:
                topic_words = model.get_topic(topic_id)
                topic_words_str = ' '.join([word for word, _ in topic_words])
                classifications[tweet_id] = topic_words_str
            else:
                classifications[tweet_id] = None
    json.dump(classifications, file, indent=4)
    file.close()

# helper
def get_all_tweets(db: str):
    all_tweets = {}
    original_tweet_collection = MONGO_CLIENT[db]["original_tweets"]
    original_tweet_df = pd.DataFrame(list(original_tweet_collection.find()))
    original_tweet_df = original_tweet_df.drop_duplicates(['id'])

    retweet_in_collection = MONGO_CLIENT[db]["retweets_of_in_community"]
    retweet_in_df = pd.DataFrame(list(retweet_in_collection.find()))
    retweet_in_df = retweet_in_df.drop_duplicates(['id'])
    retweet_in_df.loc[:, 'id'] = retweet_in_df['id'].astype(int)

    retweet_out_collection = MONGO_CLIENT[db]["retweets_of_out_community_by_in_community"]
    retweet_out_df = pd.DataFrame(list(retweet_out_collection.find()))
    retweet_out_df = retweet_out_df.drop_duplicates(['id'])
    retweet_out_df.loc[:, 'id'] = retweet_out_df['id'].astype(int)

    all_tweets["og"] = original_tweet_df
    all_tweets["retweet_in"] = retweet_in_df
    all_tweets["retweet_out"] = retweet_out_df

    return all_tweets

if __name__ == "__main__":
    # all_tweets = get_core_user_tweets("ml_community_1")
    # cleaned_tweets = get_clean_tweets(all_tweets)
    # get_topics(cleaned_tweets, "ml_community_1")
    # reduce_topics(cleaned_tweets, "ml_community_1")
    get_topic_per_tweet_in("ml_community_1")