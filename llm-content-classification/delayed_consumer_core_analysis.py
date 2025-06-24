import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import red
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from io import BytesIO
from reportlab.platypus import Image, Spacer
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import random
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap, BoundaryNorm

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# first get the core user and core users' followers
def get_core_consumers(db: str, type):
    user_collection = MONGO_CLIENT[db]['user_info']
    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 0})

    if type == "old":
        consumers = [int(x) for x in core_user_item["local follower list"]]
    else:
        consumers = [x for x in core_user_item["local follower list"]]

    return consumers

def get_random_consumers(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))

    core_user_item = user_collection.find_one({"rank": 0})
    consumers = [x for x in core_user_item["local follower list"]]

    random_consumers = user_df[~user_df['userid'].isin(consumers)]
    return random_consumers['userid'].tolist()

def get_core_consumers_restricted(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    #user_df['userid'] = user_df['userid'].astype(int)

    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 0})

    consumers = core_user_item["local follower list"]
    consumer_df = user_df[user_df['userid'].isin(consumers)]
    consumer_df = consumer_df[consumer_df['local follower list'].apply(lambda x: core_user in x)]

    consumers = consumer_df['userid'].to_list()
    return consumers

def aggregate_demand_in_time_series(consumers, db, start_date, end_date, type):
    time_series = MONGO_CLIENT[db]['retweet_in_time_series_openai']
    time_series_df = pd.DataFrame(list(time_series.find()))
    if type == "old":
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

    # filter based on core user active days

    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def aggregate_demand_out_time_series(consumers, db, start_date, end_date, type):
    time_series = MONGO_CLIENT[db]['retweet_out_time_series_openai']
    time_series_df = pd.DataFrame(list(time_series.find()))
    if type == "old":
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

    # filter based on core user active days
    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_supply_time_series(db, start_date, end_date, type):
    user_collection = MONGO_CLIENT[db]['user_info']
    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['og_tweet_time_series_openai']
    og_df = pd.DataFrame(list(og_ts.find()))
    if type == "old":
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

    # filter based on core user active days
    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_demand_in_time_series(db, start_date, end_date, type):
    user_collection = MONGO_CLIENT[db]['user_info']
    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_in_time_series_openai']
    og_df = pd.DataFrame(list(og_ts.find()))
    if type == "old":
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
    
    # filter based on core user active days
    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_core_demand_out_time_series(db, start_date, end_date, type):
    user_collection = MONGO_CLIENT[db]['user_info']
    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_out_time_series_openai']
    og_df = pd.DataFrame(list(og_ts.find()))
    if type == "old":
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
    
    # filter based on core user active days
    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df

def get_ts_window(time_series, delta):
    time_series = time_series.sort_index()
    result_data = {}
    
    for column in time_series.columns:
        column_sums = []
        center_dates = []
        
        for i in range(delta, len(time_series) - delta):
            window_sum = time_series.iloc[i - delta:i + delta + 1][column].sum()
            column_sums.append(window_sum)
            center_dates.append(time_series.index[i])
        
        result_data[column] = column_sums

    result_df = pd.DataFrame(result_data, index=center_dates)
    result_df.index.name = 'date'
    return result_df

def granger_analysis(consumer_demand_ts, core_supply_ts, db, type, k, consumer_index, core_index, lags=7):
    earliest_date = min(consumer_demand_ts.index.min(), core_supply_ts.index.min())
    latest_date = max(consumer_demand_ts.index.max(), core_supply_ts.index.max())

    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    
    consumer_demand_ts_aligned = consumer_demand_ts.reindex(full_date_range, fill_value=0)
    core_supply_ts_aligned = core_supply_ts.reindex(full_date_range, fill_value=0)

    consumer_demand_sums = consumer_demand_ts_aligned.sum().sort_values(ascending=False)
    core_supply_sums = core_supply_ts_aligned.sum().sort_values(ascending=False)
    # print(consumer_demand_sums.sum())
    # print(core_supply_sums.sum())
    # exit(0)

    with open(f"{db}/{db}_consumer.json", 'r') as f:
        consumer_topics = json.load(f)
    
    with open(f"{db}/{db}_core.json", 'r') as f:
        core_topics = json.load(f) 
    
    if type == "mix":
        consumer_series = consumer_demand_ts_aligned[consumer_topics[consumer_index]]
        core_series = core_supply_ts_aligned[core_topics[core_index]]

    if type == "top_consumer":
        for topic in consumer_topics:
            if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
                consumer_series = consumer_demand_ts_aligned[topic]
                core_series = core_supply_ts_aligned[topic]
                break
    if type == "top2":
        for topic in consumer_topics[1:]:
            if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
                consumer_series = consumer_demand_ts_aligned[topic]
                core_series = core_supply_ts_aligned[topic]
                break
    if type == "top3":
        for topic in consumer_topics[2:]:
            if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
                consumer_series = consumer_demand_ts_aligned[topic]
                core_series = core_supply_ts_aligned[topic]
                break
    if type == "top4":
        for topic in consumer_topics[3:]:
            if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
                consumer_series = consumer_demand_ts_aligned[topic]
                core_series = core_supply_ts_aligned[topic]
                break
    if type == "top5":
        for topic in consumer_topics[4:]:
            if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
                consumer_series = consumer_demand_ts_aligned[topic]
                core_series = core_supply_ts_aligned[topic]
                break

    # if type == "top_core":
    #     for topic in core_topics:
    #         if (topic in consumer_demand_ts_aligned.columns) and (topic in core_supply_ts_aligned.columns):
    #             consumer_series = consumer_demand_ts_aligned[topic]
    #             core_series = core_supply_ts_aligned[topic]
        
    # if type == "top_5_core":
    #     top_5_core_supply = core_supply_sums.nlargest(5)

    #     consumer_series = consumer_demand_ts_aligned[top_5_core_supply.index]
    #     core_series = core_supply_ts_aligned[top_5_core_supply.index]

    #     consumer_series = consumer_series.sum(axis=1)
    #     core_series = core_series.sum(axis=1)
    
    if type == "top_5_consumer":
        top_5_consumer_demand = consumer_demand_sums.nlargest(5)

        consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
        core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

        consumer_series = consumer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)
    
    # if type == "top12":
    #     top_5_consumer_demand = consumer_demand_sums.nlargest(2)

    #     consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
    #     core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

    #     consumer_series = consumer_series.sum(axis=1)
    #     core_series = core_series.sum(axis=1)
    
    # if type == "top123":
    #     top_5_consumer_demand = consumer_demand_sums.nlargest(3)

    #     consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
    #     core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

    #     consumer_series = consumer_series.sum(axis=1)
    #     core_series = core_series.sum(axis=1)

    # if type == "top1234":
    #     top_5_consumer_demand = consumer_demand_sums.nlargest(4)

    #     consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
    #     core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

    #     consumer_series = consumer_series.sum(axis=1)
    #     core_series = core_series.sum(axis=1)

    if consumer_series.empty or core_series.empty:
        return None, None
    if consumer_series.nunique() <= 1 or core_series.nunique() <= 1:
        return None, None
    # !----------------------------Core drive consumer-----------------------------------------------
    core_series = core_series.shift(k).dropna()
    common_index = core_series.index.intersection(consumer_series.index)
    common_index = common_index[:] # you can change this to change the length of the series
    core_series = core_series.loc[common_index]
    consumer_series = consumer_series.loc[common_index]

    result_consumer = adfuller(consumer_series)[1]
    result_core = adfuller(core_series)[1]

    if result_consumer > 0.05 and result_core > 0.05:
        # --------------Both not stationary---------------------------
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff= consumer_series.diff().dropna()

    if result_consumer < 0.05 and result_core > 0.05:
        # ---------------Core not stationary-------------------------
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff = consumer_series.iloc[1:]

    if result_consumer > 0.05 and result_core < 0.05:
    # --------------Consumer not stationary-----------------------
        consumer_series_diff = consumer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_consumer < 0.05 and result_core < 0.05:
        consumer_series_diff = consumer_series
        core_series_diff = core_series

    data = pd.concat([consumer_series_diff, core_series_diff], axis=1)
    data.columns = ['Consumer_Series', 'Core_Series']
    data = data.dropna()

    # ! ----------------------Random time series----------------------------------------------------
    # * here we are just generating random time series to see if the results is what we want -> for debugging purposes
    # core_series_random = core_series_diff.sample(frac=1, random_state=42).reset_index(drop=True)
    # consumer_series_ordered = consumer_series_diff.reset_index(drop=True)
    # min_len = min(len(core_series_random), len(consumer_series_ordered))
    # core_series_random = core_series_random.iloc[:min_len]
    # consumer_series_ordered = consumer_series_ordered.iloc[:min_len]
    # data_random_core = pd.concat([
    #     consumer_series_ordered.rename("Consumer_Series"),
    #     core_series_random.rename("Core_Series")
    # ], axis=1)

    # consumer_series_random = consumer_series_diff.sample(frac=1, random_state=42).reset_index(drop=True)
    # core_series_ordered = core_series_diff.reset_index(drop=True)
    # min_len = min(len(consumer_series_random), len(core_series_ordered))
    # consumer_series_random = consumer_series_random.iloc[:min_len]
    # core_series_ordered = core_series_ordered.iloc[:min_len]
    # data_random_consumer = pd.concat([
    #     consumer_series_random.rename("Consumer_Series"),
    #     core_series_ordered.rename("Core_Series")
    # ], axis=1)

    # idx = consumer_series_diff.index
    # random_consumer = pd.Series(
    #     np.random.randint(0, 21, size=270),
    #     index=idx[:270],
    #     name="random_consumer"
    # )
    # random_core = pd.Series(
    #     np.random.randint(0, 21, size=270),
    #     index=idx[:270],
    #     name="random_core"
    # )
    # def ensure_stationary(series, name):
    #     result = adfuller(series.dropna())
    #     p_value = result[1]
    #     print(f"{name} ADF p-value: {p_value:.5f}")
        
    #     if p_value > 0.05:
    #         print(f"{name} is non-stationary — differencing applied.")
    #         series = series.diff().dropna()
    #     else:
    #         print(f"{name} is already stationary.")
        
    #     return series

    #! consumer is random 
    # making one of the series random to test the code again
    # common_index = core_series.index.intersection(random_consumer.index)
    # core_series = core_series.loc[common_index] 
    # consumer_stationary = ensure_stationary(random_consumer, "random_consumer")
    # core_stationary = ensure_stationary(random_core, "random_core")
    # data_random_consumer = pd.concat([consumer_stationary, core_stationary], axis=1)
    # data_random_consumer.columns = ['Consumer_Series', 'Core_Series']
    # data_random_consumer = data_random_consumer.dropna()

    # # ! ----------------------Time series plot generation-----------------------------------------
    # used for generating the plot to visualize the time series to understand the trend and the zeros in the time series
    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # axs[0].plot(core_series_diff.index, core_series_diff, marker='o', linestyle='-')
    # axs[0].set_title(f"{type} Core Time Series Daily Counts")
    # axs[0].set_xlabel("Date")
    # axs[0].set_ylabel("Count")
    # axs[0].tick_params(axis='x', rotation=45)

    # axs[1].plot(consumer_series_diff.index, consumer_series_diff, marker='o', linestyle='-')
    # axs[1].set_title(f"{type} Consumer Time Series Daily Counts")
    # axs[1].set_xlabel("Date")
    # axs[1].set_ylabel("Count")
    # axs[1].tick_params(axis='x', rotation=45)

    # plt.tight_layout()

    # plt.savefig(f"{db}_{type}_counts.pdf", format='pdf')
    core_to_consumer_p_values = []
    try:
        granger_test_results1 = grangercausalitytests(data[['Consumer_Series', 'Core_Series']], maxlag=lags)

        for lag in range(1, lags + 1):
            test_result = granger_test_results1[lag][0]['ssr_ftest']
            p_value = test_result[1]
            core_to_consumer_p_values.append(p_value)
        
        avg_p_value_core_to_consumer = np.mean(core_to_consumer_p_values)
    except Exception as e:
        print(f"Granger test failed: {e}") 
        for lag in range(1, lags+1):
            core_to_consumer_p_values.append(-1)
        avg_p_value_core_to_consumer = -1
    # elements.append(Paragraph(f"Average p-value for Core -> Consumer: {avg_p_value_core_to_consumer:.6f}", styles["Normal"]))    
    
    # !---------------------------------------- Consumer drive core ----------------------------------------------
    consumer_series = consumer_series.shift(k).dropna()
    common_index = consumer_series.index.intersection(core_series.index)
    common_index = common_index[:] # you can change this to change the length of the series
    consumer_series = consumer_series.loc[common_index]
    core_series = core_series.loc[common_index]

    result_consumer = adfuller(consumer_series)[1]
    result_core = adfuller(core_series)[1]

    if result_consumer > 0.05 and result_core > 0.05:
        # --------------Both not stationary---------------------------
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff= consumer_series.diff().dropna()

    if result_consumer < 0.05 and result_core > 0.05:
        # ---------------Core not stationary-------------------------
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff = consumer_series.iloc[1:]

    if result_consumer > 0.05 and result_core < 0.05:
    # --------------Consumer not stationary-----------------------
        consumer_series_diff = consumer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_consumer < 0.05 and result_core < 0.05:
        consumer_series_diff = consumer_series
        core_series_diff = core_series
    
    data = pd.concat([consumer_series_diff, core_series_diff], axis=1)
    data.columns = ['Consumer_Series', 'Core_Series']
    data = data.dropna()
    # !core is random
    # making one series random to test the code again
    # common_index = consumer_series.index.intersection(random_core.index)
    # consumer_series = consumer_series.loc[common_index] 
    # consumer_stationary = ensure_stationary(random_consumer, "random_consumer")
    # core_stationary = ensure_stationary(random_core, "random_core")
    # data_random_core = pd.concat([consumer_stationary, core_stationary], axis=1)
    # data_random_core.columns = ['Consumer_Series', 'Core_Series']
    # data_random_core = data_random_core.dropna()
    consumer_to_core_p_values = []
    try:
        granger_test_results2 = grangercausalitytests(data[['Core_Series', 'Consumer_Series']], maxlag=lags)

        for lag in range(1, lags + 1):
            test_result = granger_test_results2[lag][0]['ssr_ftest']
            p_value = test_result[1]
            consumer_to_core_p_values.append(p_value)
        
        avg_p_value_consumer_to_core = np.mean(consumer_to_core_p_values)
    except:
        for lag in range(1, lags+1):
            consumer_to_core_p_values.append(-1)
        avg_p_value_consumer_to_core = -1

    p_value_ratio = avg_p_value_core_to_consumer / avg_p_value_consumer_to_core
 
    return core_to_consumer_p_values, consumer_to_core_p_values

def plot_heatmaps_v2(p_value_per_type, k):
    print(p_value_per_type)
    type_keys = list(p_value_per_type.keys())
    
    first_type_key = type_keys[0]
    deltas_in_first_type = list(p_value_per_type[first_type_key].keys())
    
    first_delta = deltas_in_first_type[0]
    communities_in_first_delta = list(p_value_per_type[first_type_key][first_delta].keys())
    
    lag_labels = [str(i) for i in range(1, 8)]
    
    for community in communities_in_first_delta:
        for tkey in type_keys:
            M_core = np.zeros((7, 2))
            M_consumer = np.zeros((7, 2))

            for col, delta_str in enumerate(deltas_in_first_type):
                community_dict = p_value_per_type.get(tkey, {}).get(delta_str, {}).get(community, {})
                
                core_vals = community_dict.get("Core drive Consumer", [])
                consumer_vals = community_dict.get("Consumer drive Core", [])
                
                for row in range(7):
                    val_core = core_vals[row] if row < len(core_vals) else np.nan
                    val_consumer = consumer_vals[row] if row < len(consumer_vals) else np.nan
                    M_core[row, col] = val_core
                    M_consumer[row, col] = val_consumer

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # *  --------------histogram ----------------------------------
            data = M_core.flatten()
            sns.histplot(
                data,
                bins=100, 
                kde=True,  
                color="skyblue",
                edgecolor="black",
                ax=axes[0]  
            )
            axes[0].set_title(f"{community} - {tkey}\nCore → Consumer")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Frequency")
            axes[0].axvline(0.05, color='red', linestyle='--', label='p = 0.05')
            axes[0].legend()

        # * --------------  1. can be used for blacking out insignificant values
            # masked_core = np.where(M_core > 0.05, np.nan, M_core)
            # masked_consumer = np.where(M_consumer > 0.05, np.nan, M_consumer)
            # # plot the heatmap with masked values
            # sns.heatmap(
            #     masked_core,
            #     cmap="YlGnBu",
            #     annot=M_core,  # show full values
            #     fmt=".5f",
            #     annot_kws={"size": 8, "color": "black"},
            #     ax=axes[0],
            #     xticklabels=deltas_in_first_type,
            #     yticklabels=lag_labels,
            #     cbar=True,
            #     linewidths=0.5,
            #     linecolor='grey'
            # )
        # !------- red below 0.05
            # cmap = ListedColormap(['red','white'])
            # # boundaries: [0–0.05) → red, [0.05–1] → white
            # norm = BoundaryNorm([0, 0.05, 1.0], cmap.N)

            # sns.heatmap(
            #     M_core,
            #     cmap=cmap,
            #     norm=norm,
            #     annot=M_core,          # show every value
            #     fmt=".5f",
            #     annot_kws={"size": 8, "color": "black"},
            #     xticklabels=deltas_in_first_type,
            #     yticklabels=lag_labels,
            #     cbar=False,            # no need for a 2-color bar
            #     linewidths=0.5,
            #     linecolor="grey",
            #     ax=axes[0]
            # )
        # !-------------------------- 2. for black patches, tied to step 1-------------------------------------
            # overlay black boxes for values > 0.05
            # for i in range(M_core.shape[0]):
            #     for j in range(M_core.shape[1]):
            #         if M_core[i, j] > 0.05:
            #             axes[0].add_patch(Rectangle((j, i), 1, 1, color='black'))

            # axes[0].set_title(f"{community} - {tkey}\nCore → Consumer")
            # axes[0].set_xlabel("Window Size (Δ)")
            # axes[0].set_ylabel("Lag")

            # ! 1. Repeat for consumer matrix, blackout
            # sns.heatmap(
            #     masked_consumer,
            #     cmap="YlGnBu",
            #     annot=M_consumer,
            #     fmt=".5f",
            #     annot_kws={"size": 8, "color": "black"},
            #     ax=axes[1],
            #     xticklabels=deltas_in_first_type,
            #     yticklabels=lag_labels,
            #     cbar=True,
            #     linewidths=0.5,
            #     linecolor='grey'
            # )
            # *  --------------histogram ----------------------------------
            data = M_consumer.flatten()
            sns.histplot(
                data,
                bins=100, 
                kde=True,  
                color="skyblue",
                edgecolor="black",
                ax=axes[1]  
            )
            axes[1].set_title(f"{community} - {tkey}\nConsumer → Core")
            axes[1].set_xlabel("Value")
            axes[1].set_ylabel("Frequency")
            axes[1].axvline(0.05, color='red', linestyle='--', label='p = 0.05')
            axes[1].legend()

            # * --------- for red below 0.05 --------------------
            # sns.heatmap(
            #     M_consumer,
            #     cmap=cmap,
            #     norm=norm,
            #     annot=M_consumer,          # show every value
            #     fmt=".5f",
            #     annot_kws={"size": 8, "color": "black"},
            #     xticklabels=deltas_in_first_type,
            #     yticklabels=lag_labels,
            #     cbar=False,            # no need for a 2-color bar
            #     linewidths=0.5,
            #     linecolor="grey",
            #     ax=axes[1]
            # )

            # !-------------------------- 2. for black patches, tied to step 1-------------------------------------
            # for i in range(M_consumer.shape[0]):
            #     for j in range(M_consumer.shape[1]):
            #         if M_consumer[i, j] > 0.05:
            #             axes[1].add_patch(Rectangle((j, i), 1, 1, color='black'))

            # axes[1].set_title(f"{community} - {tkey}\nConsumer → Core")
            # axes[1].set_xlabel("Window Size (Δ)")
            # axes[1].set_ylabel("Lag")

            # add a supertitle
            plt.suptitle(f"P-value Heatmaps (Lag vs. Δ)\nCommunity: {community}, Type: {tkey}, K: {k}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # save figure
            file_name = f"single_k123/k{k}_{community}_{tkey}_heatmap.pdf".replace(" ", "_")
            plt.savefig(file_name, dpi=300, format='pdf')
            plt.close(fig)

    print("Heatmaps for each (type, community) combination were generated and saved.")

def calculate_avg(p_value_per_type):
    result = {}

    for top_level_type, delta_dict in p_value_per_type.items():
        # e.g. "top_core", "top_consumer", etc.
        result[top_level_type] = {}
        
        for delta, communities_dict in delta_dict.items():
            # e.g. delta = "1", "2", "3", etc.
            
            # We'll accumulate sums and counts across all communities
            cdc_sums = []  # sums for "Core drive Consumer"
            cdc_count = 0
            c2c_sums = []  # sums for "Consumer drive Core"
            c2c_count = 0
            
            for community_name, directions_dict in communities_dict.items():
                # directions_dict is something like:
                # {
                #   "Core drive Consumer": [...],
                #   "Consumer drive Core": [...]
                # }
                # 1) "Core drive Consumer"
                cdc_values = directions_dict["Core drive Consumer"]
                if not cdc_sums:
                    cdc_sums = [0.0] * len(cdc_values)
                for i, val in enumerate(cdc_values):
                    cdc_sums[i] += val
                cdc_count += 1

                # 2) "Consumer drive Core"
                c2c_values = directions_dict["Consumer drive Core"]
                if not c2c_sums:
                    c2c_sums = [0.0] * len(c2c_values)
                for i, val in enumerate(c2c_values):
                    c2c_sums[i] += val
                c2c_count += 1
            # Compute averages
            if cdc_count > 0:
                cdc_avgs = [s / cdc_count for s in cdc_sums]
            else:
                cdc_avgs = []
            
            if c2c_count > 0:
                c2c_avgs = [s / c2c_count for s in c2c_sums]
            else:
                c2c_avgs = []
            
            # Store in the requested structure:
            # top_core -> "1" -> "all_community" -> "Core drive Consumer": [...]
            result[top_level_type].setdefault(delta, {})
            result[top_level_type][delta]["all_community"] = {
                "Core drive Consumer": cdc_avgs,
                "Consumer drive Core": c2c_avgs
            }

    return result

if __name__ == "__main__":
    communities = ["astronomy_expanded", "comics_expanded"]
    community_date_ranges = {
    "ml_community_1": {"start_date": "2022-04-16", "end_date": "2023-03-07"},
    "new_all_tweets_ml_community_2": {"start_date": "2022-04-16", "end_date": "2023-03-07"},
    "rachel_chess_community": {"start_date": "2021-01-02", "end_date": "2022-01-01"},
    "astronomy": {"start_date": "2023-06-19", "end_date": "2024-07-14"},
    "climate_change": {"start_date": "2023-08-03", "end_date": "2024-06-24"}, 
    "comics": {"start_date": "2024-02-12", "end_date": "2024-07-01"},  
    "history": {"start_date": "2023-07-30", "end_date": "2024-08-20"},
    "microbiology": {"start_date": "2023-08-17", "end_date": "2024-08-21"}, 
    "neuroscience": {"start_date": "2023-07-24", "end_date": "2024-07-15"}, 
    "poetry": {"start_date": "2023-07-16", "end_date": "2024-08-23"}, 
    "history_expanded": {"start_date": "2024-09-01", "end_date": "2025-02-26"}, 
    "comics_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "astronomy_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "game_development": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "urban_planning": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "math": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "poetry_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "climate_change_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "neuroscience_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "science_fiction_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}}

    types = ["mix"]
    # types = {
    #         "astronomy_expanded": ["top4"], 
    #         "climate_change_expanded": ["top3"], 
    #         "comics_expanded": ["top3"], 
    #         "game_development": ["top3"], 
    #         "math": ["top_consumer"], 
    #         "poetry_expanded": ["top4"]
    #     }
    p_value_per_type = {}
    p_value_per_comm  ={comm: [] for comm in communities}
    old = ["ml_community_1", "new_all_tweets_ml_community_2", "rachel_chess_community",]

    for k in range(1, 10):
        for community in communities:
            if community in old:
                consumers = get_core_consumers(community, "old")
                # consumers = get_random_consumers(community)
                date_range = community_date_ranges[community]
                consumer_demand_in_ts = aggregate_demand_in_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
                consumer_demand_out_ts = aggregate_demand_out_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")

                core_supply_ts = get_core_supply_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
                core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
                core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
            else:
                consumers = get_core_consumers(community, "new")
                # consumers = get_random_consumers(community)
                date_range = community_date_ranges[community]
                consumer_demand_in_ts = aggregate_demand_in_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
                consumer_demand_out_ts = aggregate_demand_out_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")

                core_supply_ts = get_core_supply_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
                core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
                core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")

            # NOTE: combine core demand and supply
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

            # NOTE: combine consumer demand in and out
            # ----------------------------------------------------------------
            consumer_ts = pd.concat([consumer_demand_in_ts, consumer_demand_out_ts])
            # consumer_ts = consumer_demand_in_ts
            consumer_ts = consumer_ts.reset_index()
            consumer_ts = consumer_ts.groupby('date', as_index=False).sum()
            consumer_ts = consumer_ts.set_index('date')
            # ----------------------------------------------------------------
            with open(f"{community}/{community}_consumer.json", 'r') as file:
                consumer_topics = json.load(file)

            for consumer_index in range(len(consumer_topics)):
                for core_index in range(len(consumer_topics)):
                    if consumer_index != core_index:
                        for delta in range(1, 8):
                            temp_core_ts = get_ts_window(core_ts, delta)
                            temp_consumer_ts = get_ts_window(consumer_ts, delta)
                            for type in types:
                                key_name = f"{type}"
                                # granger_analysis(consumer_ts, core_ts, community, type=type, k=k)
                                core_to_consumer_p, consumer_to_core_p = granger_analysis(temp_consumer_ts, temp_core_ts, community, type=type, consumer_index=consumer_index, core_index=core_index, k=k)
                                if core_to_consumer_p is None:
                                    continue
                                p_value_per_comm[community].append(core_to_consumer_p)
                        if key_name not in p_value_per_type:
                            p_value_per_type[key_name] = {}
                        if delta not in p_value_per_type[key_name]:
                            p_value_per_type[key_name][delta] = {}
                        else:
                            if community not in p_value_per_type[key_name][delta]:
                                p_value_per_type[key_name][delta][community] = {}

                        p_value_per_type[key_name][delta][community] = {
                            "Core drive Consumer": core_to_consumer_p,
                            "Consumer drive Core": consumer_to_core_p
                        }

        # # result = calculate_avg(p_value_per_type)
        # print(p_value_per_type)
        # ! uncomment below if you want to save the results into a json, you can also change the code to save it to the database
        # with open(f"r0/game_development_consumer_core_{k}.json", "w") as file:
        #     json.dump(p_value_per_type, file)
        plot_heatmaps_v2(p_value_per_type, k)  
    # ! below is for the histogram for overall trend
    # for comm, vals in p_value_per_comm.items():
    #     plt.figure()                        # new figure for each community
    #     plt.hist(vals)                      # histogram of the p-values
    #     plt.title(f"P Distribution for {comm}")
    #     plt.xlabel("P value")
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.savefig(f"all_comb/{comm}_histogram.png")  # save to file
    #     plt.close()    
    # ! below is for generating the pdf to see all the values, need to uncomment all the blocks with 'element' in the file
    # pdf_file = f"consumer_core_analysis_k4.pdf"
    # doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter))
    # styles = getSampleStyleSheet()
    # elements = []
    # custom_style = ParagraphStyle(
    #     'CustomStyle',
    #     parent=styles['Normal'],
    #     textColor=red
    # )

    # for type in types:
    #     table_data = [["Delta", "Relationship"] + communities]
    #     for delta, communities_data in p_value_per_type[type].items():
    #         for relationship in ["Core drive Consumer", "Consumer drive Core", "P Value Ratio"]:
    #             row = [delta, relationship]
    #             for community in communities:
    #                 p_value = communities_data.get(community, {}).get(relationship, "-")
    #                 cell_value = f"{p_value:.6f}" if isinstance(p_value, (int, float)) and p_value is not None else "-"
    #                 if relationship == "P Value Ratio" and isinstance(p_value, (int, float)) and p_value > 1:
    #                     formatted_value = f"<font color='red'>{cell_value}</font>"
    #                 else:
    #                     formatted_value = cell_value
    #                 row.append(Paragraph(formatted_value, styles["Normal"]))
    #             table_data.append(row)

        # table = Table(table_data)
        # style = TableStyle([
        # ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  
        # ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), 
        # ('WORDWRAP', (0, 0), (-1, -1)),  
        # ('FONTSIZE', (0, 0), (-1, -1), 8),
        # ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        # ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        # ('GRID', (0, 0), (-1, -1), 1, colors.black)
        # ])

        # # alternate row colors for readability
        # for idx, row in enumerate(table_data[1:], start=1):
        #     bg_color = colors.lightgrey if row[0] % 2 == 0 else colors.whitesmoke
        #     style.add('BACKGROUND', (0, idx), (-1, idx), bg_color)

        # table.setStyle(style)

        # elements.append(PageBreak())
        # elements.append(Paragraph(f"Granger Causality Analysis Results for {type}", styles["Heading2"]))
        # elements.append(table)

    #     for relationship in ["Core drive Consumer", "Consumer drive Core", "P Value Ratio"]:
    #         plt.figure(figsize=(8, 5))
    #         plt.xlabel('Delta')
    #         plt.ylabel('P-Value')
    #         plt.title(f'{type} - {relationship} Across Communities')

    #         for community in communities:
    #             deltas = [delta for delta in range(1, 8) if delta in p_value_per_type[type] and community in p_value_per_type[type][delta]]
    #             p_values = [p_value_per_type[type][delta][community].get(relationship, None) for delta in deltas]

    #             if deltas and any(p_values):
    #                 plt.plot(deltas, p_values, marker='o', linestyle='-', label=community)

    #         plt.legend(title="Communities", loc="best")
    #         plt.grid(True)
    #         plt.savefig(f"{type}_{relationship.replace(' ', '_')}.png")
    #         plt.close()

    # doc.build(elements)




    