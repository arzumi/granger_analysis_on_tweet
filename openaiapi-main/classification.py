from openai import OpenAI
import pandas as pd
import json
from decimal import Decimal


# Tweet File Path
json_file_path = "/Users/ptran/Downloads/Final Report Communities/new_rachel_chess_content_market/retweets_of_in_community.json"
df = pd.read_json(json_file_path, lines=True)
tweet_data = df[['id', 'text']]

# Read the topics from the file into a string
input_topics_file_path = "/Users/ptran/Downloads/topics_data/chess_merge_basic_10.txt"
with open(input_topics_file_path, "r") as file:
    topics_string = file.read()

# Output file path to save the topic classifications
output_file_path = "/Users/ptran/Downloads/topics_data/chess_classification_retweets_of_in_community"
file = open(output_file_path, 'w')

system_instruction = f"""
Classify the following input text into one of the following categories: 
{topics_string}
Be sure to state only the topic without the number.
The desired output format:
Topic: xxx
"""

i = 0
classifications = {}

# Iterate over each tweet in the DataFrame
for index, row in tweet_data.iloc[0:].iterrows():

    # Extract tweet text and tweet id
    id_value = row['id']['$numberDouble']
    tweet_text = row['text']

    # Convert the tweet ID to an integer
    tweet_id = int(Decimal(id_value))
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": tweet_text}
    ]

    # Make a request to the OpenAI API to classify the tweet
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.5
    )

    topic = response.choices[0].message.content.strip()

    try:
        topic_text = topic.split(": ")[1]
    except IndexError:
        print("IndexError")
        topic_text = "Chess Literature and Media"       # Default to this topic in case of error


    # Store the classification in a dictionary
    classifications[tweet_id] = topic_text

    if i % 100 == 0:
        print(i)

    i +=1

with open(output_file_path, 'w') as output_file:
    json.dump(classifications, output_file, indent=4)

file.close()
