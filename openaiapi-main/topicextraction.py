from openai import OpenAI
import pandas as pd

# Batch size for feeding in tweets
BATCH_SIZE = 20


# Tweet File Path
json_file_path = "/Users/ptran/Downloads/Final Report Communities/new_rachel_chess_content_market/original_tweets.json"
df = pd.read_json(json_file_path, lines=True)
tweets_text = df['text']
NUM_TWEETS = len(df)

# Output file path to save the topics
output_file_path = "/Users/ptran/Downloads/topics_data/chess_topics_basic.txt"
file = open(output_file_path, 'w')

system_instruction = f"""
Read the text below and list up to 3 topics. Each topic should contain fewer than 3 words. Ensure you only return the topic and nothing more.
The desired output format:
Topic 1: xxx\nTopic 2: xxx\nTopic 3: xxx
"""

unique_topics = set()

batch_iter = 0
num_iters = (NUM_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE

# Process tweets in batches
while batch_iter < num_iters and batch_iter * BATCH_SIZE <= NUM_TWEETS:
    text_input = ""

    # Collect tweets for the current batch
    for i in range(batch_iter * BATCH_SIZE, min((batch_iter + 1) * BATCH_SIZE, NUM_TWEETS)):
        text_input += f"Tweet {i + 1}: {tweets_text.iloc[i]}\n"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": text_input}
    ]

    # Make a request to the OpenAI API to generate topics from the tweets
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.5
    )

    # Extract the topics from the API response
    topics = response.choices[0].message.content.strip().split('\n')

    # Add each extracted topic to the set of unique topics
    for topic in topics:
        topic_text = topic.split(": ")[1]
        unique_topics.add(topic_text)

    batch_iter += 1

for topic in unique_topics:
    file.write(f"{topic}\n")

file.close()