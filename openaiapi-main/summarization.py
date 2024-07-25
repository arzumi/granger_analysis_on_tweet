from openai import OpenAI
import pandas as pd


file_path = "/Users/ptran/Downloads/topics_data/chess_topics_basic.txt"
with open(file_path, "r") as file:
    topics_string = file.read()

output_file_path = "/Users/ptran/Downloads/topics_data/chess_merge_basic_10.txt"
file = open(output_file_path, 'w')

system_instruction = f"""
Summarize and merge the following list of topics into 10 final topics
"""

messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": topics_string}
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=200,
    temperature=0.5
)

topic = response.choices[0].message.content.strip()

with open(output_file_path, 'w') as output_file:
    output_file.write(topics)

# Close the file
file.close()