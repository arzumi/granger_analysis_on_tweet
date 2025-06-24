# Intro
- This README talks about how to import a new community into the database
- You need the following file:
     - import.py

# First step - get all users
- Using the community detection algorithm, you can get all the users in a given community. 
    - The user list should be a csv file, and you use directly use MongoDB's interface to import the user list. 
# Second step - import all tweets
- Once you download the tweets using *SNACE.py* file, you will get a json file with all the tweets. Use the json file and the *import_tweets* function to import all the tweets into the database. 
# Third step - import followers and followings
- Get the friends list of all users, you can ask Alex about this, and use the json file to import all the local followers and followings. 
    - Use *import_followings* and *import_followers* functions, no need to run the filter. The filter is used for previous way of importing friends. 

# At the end you should have 3 databases for 3 different tweet types, and a user database that has all of user's information including their followers, following, rank, userid and etc. 