import json
import numpy as np
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# first get the core user and core users' followers
def get_core_consumers(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})

    consumers = core_user_item["local follower list"]
    return consumers

def get_core_consumers_restricted(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    user_df['userid'] = user_df['userid'].astype(int)

    core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})

    consumers = core_user_item["local follower list"]
    consumer_df = user_df[user_df['userid'].isin(consumers)]
    consumer_df = consumer_df[consumer_df['local follower list'].apply(lambda x: core_user in x)]

    consumers = consumer_df['userid'].to_list()
    return consumers

def aggregate_demand_in_time_series(consumers, db):
    time_series = MONGO_CLIENT[db]['retweet_in_time_series_bert']
    time_series_df = pd.DataFrame(list(time_series.find()))
    time_series_df['user_id'] = time_series_df['user_id'].astype(int)

    time_series_df = time_series_df[time_series_df['user_id'].isin(consumers)]

    aggregated_time_series = []
    for series in time_series_df['time_series']:
        aggregated_time_series.extend(series)
    
    aggregated_df = pd.DataFrame(aggregated_time_series)
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by='date').reset_index(drop=True)

    topics_df = aggregated_df.set_index('date')['topics'].apply(pd.Series).fillna(0)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df = topics_df.sort_index()
    
    start_date = "2022-04-15"
    end_date = "2023-01-12" 

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def aggregate_demand_out_time_series(consumers, db):
    time_series = MONGO_CLIENT[db]['retweet_out_time_series_bert']
    time_series_df = pd.DataFrame(list(time_series.find()))
    time_series_df['user_id'] = time_series_df['user_id'].astype(int)

    time_series_df = time_series_df[time_series_df['user_id'].isin(consumers)]

    aggregated_time_series = []
    for series in time_series_df['time_series']:
        aggregated_time_series.extend(series)
    
    aggregated_df = pd.DataFrame(aggregated_time_series)
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by='date').reset_index(drop=True)

    topics_df = aggregated_df.set_index('date')['topics'].apply(pd.Series).fillna(0)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df = topics_df.sort_index()
    
    start_date = "2022-04-15"
    end_date = "2023-01-12" 

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_supply_time_series(db):
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['og_tweet_time_series_bert']
    og_df = pd.DataFrame(list(og_ts.find()))
    og_df['user_id'] = og_df['user_id'].astype(int)
    og_df = og_df[og_df['user_id'] == core_user]

    aggregated_time_series = []
    for series in og_df['time_series']:
        aggregated_time_series.extend(series)
    
    aggregated_df = pd.DataFrame(aggregated_time_series)
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by='date').reset_index(drop=True)
    topics_df = aggregated_df.set_index('date')['topics'].apply(pd.Series).fillna(0)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df = topics_df.sort_index()
    
    start_date = "2022-04-15"
    end_date = "2023-01-12" 

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_demand_in_time_series(db):
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_in_time_series_bert']
    og_df = pd.DataFrame(list(og_ts.find()))
    og_df['user_id'] = og_df['user_id'].astype(int)
    og_df = og_df[og_df['user_id'] == core_user]

    aggregated_time_series = []
    for series in og_df['time_series']:
        aggregated_time_series.extend(series)
    
    aggregated_df = pd.DataFrame(aggregated_time_series)
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by='date').reset_index(drop=True)
    topics_df = aggregated_df.set_index('date')['topics'].apply(pd.Series).fillna(0)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df = topics_df.sort_index()
    
    start_date = "2022-04-15"
    end_date = "2023-01-12" 

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_demand_out_time_series(db):
    user_collection = MONGO_CLIENT[db]['user_info']
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_out_time_series_bert']
    og_df = pd.DataFrame(list(og_ts.find()))
    og_df['user_id'] = og_df['user_id'].astype(int)
    og_df = og_df[og_df['user_id'] == core_user]

    aggregated_time_series = []
    for series in og_df['time_series']:
        aggregated_time_series.extend(series)
    
    aggregated_df = pd.DataFrame(aggregated_time_series)
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by='date').reset_index(drop=True)
    topics_df = aggregated_df.set_index('date')['topics'].apply(pd.Series).fillna(0)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df = topics_df.sort_index()

    start_date = "2022-04-15"
    end_date = "2023-01-12" 

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df


def granger_analysis(consumer_demand_ts, core_supply_ts, lags=7):
    earliest_date = min(consumer_demand_ts.index.min(), core_supply_ts.index.min())
    latest_date = max(consumer_demand_ts.index.max(), core_supply_ts.index.max())
    
    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    
    consumer_demand_ts_aligned = consumer_demand_ts.reindex(full_date_range, fill_value=0)
    core_supply_ts_aligned = core_supply_ts.reindex(full_date_range, fill_value=0)

    consumer_demand_sums = consumer_demand_ts_aligned.sum().sort_values(ascending=False)
    core_supply_sums = core_supply_ts_aligned.sum().sort_values(ascending=False)
    
    #find the one with the max topic
    # --------------------------------------------------------
    # top_consumer_demand_topic = consumer_demand_sums.idxmax()
    # top_core_supply_topic = core_supply_sums.idxmax()

    # consumer_series = consumer_demand_ts_aligned[top_core_supply_topic]
    # core_series = core_supply_ts_aligned[top_core_supply_topic]
    # ---------------------------------------------------------

    # combines top 5 topics
    # ---------------------------------------------------------
    top_5_consumer_demand = consumer_demand_sums.nlargest(5)
    top_5_core_supply = core_supply_sums.nlargest(5)

    consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
    core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

    # -----------------------------------------------------------

    consumer_series = consumer_series.sum(axis=1)
    core_series = core_series.sum(axis=1)

    consumer_zeros = (consumer_series == 0).sum()
    core_zeros = (core_series == 0).sum()
    print(f"{consumer_zeros} and the total for consumer: {len(consumer_series)}")
    print(f"{core_zeros} and the total for core: {len(core_series)}")

    result_consumer = adfuller(consumer_series)[1]
    result_core = adfuller(core_series)[1]
    print(f"{result_consumer} and core: {result_core}")

    # ---------------Core not stationary-------------------------
    # core_series_diff = core_series.diff().dropna()
    # consumer_series_aligned = consumer_series.iloc[1:]

    # --------------Both not stationary---------------------------
    core_series_diff = core_series.diff().dropna()
    consumer_series_diff= consumer_series.diff().dropna()

    # --------------Consumer not stationary-----------------------
    # consumer_series_diff = consumer_series.diff().dropna()
    # core_series_aligned = core_series.iloc[1:]

    #----------------------------Granger Analysis-----------------------------------------------
    data = pd.concat([consumer_series_diff, core_series_diff], axis=1)
    data.columns = ['Consumer_Series', 'Core_Series']
    data = data.dropna()
    granger_test_results = grangercausalitytests(data[['Consumer_Series', 'Core_Series']], maxlag=lags, verbose=True)

    for lag in range(1, lags + 1):
        test_result = granger_test_results[lag][0]['ssr_ftest']
        f_statistic = test_result[0]
        p_value = test_result[1]
        print(f"Lag {lag} | F-Statistic: {f_statistic:.6f} | P-value: {p_value:.6f}")
   


if __name__ == "__main__":
    consumers = get_core_consumers("ml_community_1")
    # consumers = get_core_consumers_restricted("ml_community_1")
    consumer_demand_in_ts = aggregate_demand_in_time_series(consumers, "ml_community_1")
    consumer_demand_out_ts = aggregate_demand_out_time_series(consumers, "ml_community_1")

    core_supply_ts = get_core_supply_time_series("ml_community_1")
    core_demand_in_ts = get_core_demand_in_time_series("ml_community_1")
    core_demand_out_ts = get_core_demand_out_time_series("ml_community_1")

    # combine core demand and supply
    # ----------------------------------------------------------------
    combined_df = pd.concat([core_supply_ts, core_demand_in_ts])
    combined_df = combined_df.reset_index()
    combined_df = combined_df.groupby('date', as_index=False).sum()
    combined_df = combined_df.set_index('date')

    core_ts = pd.concat([core_demand_out_ts, combined_df])
    core_ts = core_ts.reset_index()
    core_ts = core_ts.groupby('date', as_index=False).sum()
    core_ts = core_ts.set_index('date')
    # ----------------------------------------------------------------

    # combine consumer demand in and out
    # ----------------------------------------------------------------
    consumer_ts = pd.concat([consumer_demand_in_ts, consumer_demand_out_ts])
    consumer_ts = consumer_ts.reset_index()
    consumer_ts = consumer_ts.groupby('date', as_index=False).sum()
    consumer_ts = consumer_ts.set_index('date')
    # ----------------------------------------------------------------

    result = granger_analysis(consumer_ts, core_ts)

    # NOTE: checking for the longest continuous period
    # -----------------------------------------------------------------------------

    # df = core_supply_ts.sum(axis=1)
    # df_in = core_demand_in_ts.sum(axis=1)
    # df_out = core_demand_out_ts.sum(axis=1)

    # threshold = 2

    # # identify continuous periods where the condition is met
    # continuous_periods = df_out.rolling(window=threshold+1, min_periods=1).apply(
    #     lambda x: np.sum(x == 0) <= threshold).astype(bool)

    # #group consecutive `True` values
    # continuous_periods_group = (continuous_periods != continuous_periods.shift()).cumsum()

    # # calculate the length of each group
    # group_lengths = continuous_periods.groupby(continuous_periods_group).sum()

    # # find the group with the longest `True` sequence
    # longest_group = group_lengths.idxmax()
    # longest_length = group_lengths.max()

    # # extract the rows for the longest continuous period
    # longest_period = df_out[continuous_periods_group == longest_group]

    # print(f"Longest Continuous Period Length: {longest_length}")
    # print(longest_period.to_string())
    # -----------------------------------------------------------------------------