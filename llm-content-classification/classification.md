# Intro
- This README talks stricly about how to classify tweets, how to store the classfied tweets and the time series generation. 
- First make sure you have your openai key and add it to the constant variable at the beginning of the file. 
- You need the following files:
    - openai_content_classification.py
    - openai_time_series_generation.py
    - word_counting.py

# First step - get topics
- Before getting the topics create the following folders in the main folder (llm-content-classification) for the communities you are interested in classifying. 
    - {community name}/openai/new
- After folder create we need to get summarized topics using the core user's tweet. 
-       # all_tweets = get_core_user_tweets(community)
        # get_all_topics(all_tweets, community)
        # get_summarized_topics(community)
    - Uncomment the above lines in the for loop will give you the summarized topics for the core user

# A. Second step - classify tweets using openai
- In the for loop uncomment the below code to classify the OG tweets and retweets of out of community tweets
-       # get_topic_per_tweet_og(community)
        # get_topic_per_tweet_out(community)
- After the classification is finished you will see a json file stored in the respective folder. Now you need to restructure the json file. 
    - Add commas after each dictionary, enclose the entire json in a list
    - So at the end you should have a list of dictionaries. 
- Now run the *flatten_json* function on the two json that is generated. 
- Then run the classification function for the retweet of in community: *get_topic_per_tweet_in*
- **NOTE**: if there are lots of ignored tweets you can run the *classify_ignored_tweets* but at the end you will have to copy over the results to the flattened json file.  
- At the end you should only have 3 files where each one represent one type of tweets and the name ends with *new.json*. They will be stored in respective folders. 

# B. Second step - classify tweets using word frequency
- If you want faster classification you can also use the word frequency method. 
- Use the topics generated in step one and feed them into OpenAI interface and get 50 words per topic. 
- Replace the variable *topics* in the *word_counting.py* file with all the topics and their respective related words. 
- Run the classification function for retweets out and OG tweets. 
    - For in community retweets you can use the same classification from the *openai_classification.py*.

# Third step - time series generation
- First run the *remove_none_type* function to remove the none types on the stored json. You should have a file with name ending in *_new_clean.json*. This will be the final type of file that you will be using throughout the program. 
- Then you can run the three time series generation functions to generate the time series. 
- **NOTE**: if you are using openai method and also want to use the word frequency method to classify tweets then generate time series, it is adviced to differentiate the name of the database used to store the respective time series to avoid any confusion. 

 ## By now, you should have 3 json files with each tweet's type, and 3 databases for 3 different time series in each community.