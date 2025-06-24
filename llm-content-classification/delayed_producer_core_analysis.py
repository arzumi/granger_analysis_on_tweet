import json
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
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
import matplotlib.colors as mcolors

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# first get the core user and core users' followers
def get_producers(db: str, type):
    # ----------------------- Producers that follow the core agent -----------------
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    if type == "old":
        user_df['userid'] = user_df['userid'].astype(int)

    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
        producers = [int(x) for x in core_user_item["local following list"]]
        return producers
    else:
        core_user_item = user_collection.find_one({"rank": 6})
    core_user = core_user_item["userid"]

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 6})

    producers = core_user_item["local following list"]
    producer_df = user_df[user_df['userid'].isin(producers)]
    producer_df = producer_df[producer_df['local following list'].apply(lambda x: core_user in x)]

    producers = producer_df['userid'].to_list()
    return producers

def get_random_producers(db:str):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))

    core_user_item = user_collection.find_one({"rank": 6})
    core_user = core_user_item["userid"]
    producers = core_user_item["local following list"]

    non_core_following = user_df[~user_df['userid'].isin(producers)]
    non_core_followers = non_core_following[~non_core_following['local following list'].apply(lambda x: core_user in x)]

    return non_core_followers['userid'].tolist()


def aggregate_demand_in_time_series(producers, db, start_date, end_date, type):
    time_series = MONGO_CLIENT[db]['retweet_in_time_series_word']
    time_series_df = pd.DataFrame(list(time_series.find()))
    if type == "old":
        time_series_df['user_id'] = time_series_df['user_id'].astype(int)

    time_series_df = time_series_df[time_series_df['user_id'].isin(producers)]

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

def aggregate_demand_out_time_series(producers, db, start_date, end_date, type):
    time_series = MONGO_CLIENT[db]['retweet_out_time_series_word']
    time_series_df = pd.DataFrame(list(time_series.find()))
    if type == "old":
        time_series_df['user_id'] = time_series_df['user_id'].astype(int)

    time_series_df = time_series_df[time_series_df['user_id'].isin(producers)]

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

def aggregate_supply_time_series(producers, db, start_date, end_date, type):
    time_series = MONGO_CLIENT[db]['og_tweet_time_series_word']
    time_series_df = pd.DataFrame(list(time_series.find()))
    if type == "old":
        time_series_df['user_id'] = time_series_df['user_id'].astype(int)

    time_series_df = time_series_df[time_series_df['user_id'].isin(producers)]

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

def get_core_demand_in_time_series(db, start_date, end_date, type):
    user_collection = MONGO_CLIENT[db]['user_info']
    if db == "rachel_chess_community":
        core_user_item = user_collection.find_one({"userid": 228660231})
    else:
        core_user_item = user_collection.find_one({"rank": 6})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 6})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_in_time_series_word']
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
        core_user_item = user_collection.find_one({"rank": 6})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 6})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_out_time_series_word']
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

def granger_analysis(core_demand_ts, producer_supply_ts, db, type, k, lags=7):
    earliest_date = min(core_demand_ts.index.min(), producer_supply_ts.index.min())
    latest_date = max(core_demand_ts.index.max(), producer_supply_ts.index.max())

    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    
    core_demand_ts_aligned = core_demand_ts.reindex(full_date_range, fill_value=0)
    producer_supply_ts_aligned = producer_supply_ts.reindex(full_date_range, fill_value=0)

    producer_sums = producer_supply_ts_aligned.sum().sort_values(ascending=False)
    core_sums = core_demand_ts_aligned.sum().sort_values(ascending=False)

    # read the file that contains the consumer and core topics
    with open(f"{db}/{db}_consumer.json", 'r') as f:
        consumer_topics = json.load(f)
    
    with open(f"{db}/{db}_core.json", 'r') as f:
        core_topics = json.load(f)

    if type == "mix":
        if community == "astronomy_expanded" or community == "poetry_expanded":
            producer_series = producer_supply_ts_aligned[consumer_topics[3]]
        if community == "math":
            producer_series = producer_supply_ts_aligned[consumer_topics[0]]
        else:
            producer_series = producer_supply_ts_aligned[consumer_topics[2]]
        core_series = core_demand_ts_aligned[consumer_topics[0]]
    
    if type == "top_consumer":
        for topic in consumer_topics:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
                break
    if type == "top2":
        for topic in consumer_topics[1:]:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
                break
    if type == "top3":
        for topic in consumer_topics[2:]:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
                break
    if type == "top4":
        for topic in consumer_topics[3:]:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
                break
    if type == "top5":
        for topic in consumer_topics[4:]:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
                break
    

    if type == "top_core":
        for topic in core_topics:
            if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
                producer_series = producer_supply_ts_aligned[topic]
                core_series = core_demand_ts_aligned[topic]
    
    if type == "top_5_core":
        top_5_core_topics = core_topics[0:5]

        producer_series = producer_supply_ts_aligned[top_5_core_topics]
        core_series = core_demand_ts_aligned[top_5_core_topics]

        producer_series = producer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)

    if type == "top_5_consumer":
        top_5_consumer_topics = consumer_topics[0:5]

        producer_series = producer_supply_ts_aligned[top_5_consumer_topics]
        core_series = core_demand_ts_aligned[top_5_consumer_topics]

        producer_series = producer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)
    
    if type == "top12":
        top_5_consumer_topics = consumer_topics[0:2]

        producer_series = producer_supply_ts_aligned[top_5_consumer_topics]
        core_series = core_demand_ts_aligned[top_5_consumer_topics]

        producer_series = producer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)
    
    if type == "top123":
        top_5_consumer_topics = consumer_topics[0:3]

        producer_series = producer_supply_ts_aligned[top_5_consumer_topics]
        core_series = core_demand_ts_aligned[top_5_consumer_topics]

        producer_series = producer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)
    
    if type == "top1234":
        top_5_consumer_topics = consumer_topics[0:4]

        producer_series = producer_supply_ts_aligned[top_5_consumer_topics]
        core_series = core_demand_ts_aligned[top_5_consumer_topics]

        producer_series = producer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)

    # ! ----------------------Time series plot generation-----------------------------------------
    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # axs[0].plot(core_series.index, core_series, marker='o', linestyle='-')
    # axs[0].set_title(f"{type} Core Time Series Daily Counts")
    # axs[0].set_xlabel("Date")
    # axs[0].set_ylabel("Count")
    # axs[0].tick_params(axis='x', rotation=45)

    # axs[1].plot(producer_series.index, producer_series, marker='o', linestyle='-')
    # axs[1].set_title(f"{type} Producer Time Series Daily Counts")
    # axs[1].set_xlabel("Date")
    # axs[1].set_ylabel("Count")
    # axs[1].tick_params(axis='x', rotation=45)

    # plt.tight_layout()
    # plt.savefig(f"{db}_{type}_counts.pdf", format='pdf')

    # !----------------------------Producer drive core-----------------------------------------------
    producer_series = producer_series.shift(k).dropna()
    common_index = producer_series.index.intersection(core_series.index)
    producer_series = producer_series.loc[common_index]
    core_series = core_series.loc[common_index]

    result_core = adfuller(core_series)[1]
    result_producer = adfuller(producer_series)[1]

    if result_core > 0.05 and result_producer > 0.05:
        # --------------Both not stationary---------------------------
        core_series_diff = core_series.diff().dropna()
        producer_series_diff= producer_series.diff().dropna()

    if result_core < 0.05 and result_producer > 0.05:
        # ---------------Producer not stationary-------------------------
        producer_series_diff = producer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_core > 0.05 and result_producer < 0.05:
    # --------------Core not stationary-----------------------
        core_series_diff = core_series.diff().dropna()
        producer_series_diff = producer_series.iloc[1:]

    if result_producer < 0.05 and result_core < 0.05:
        producer_series_diff = producer_series
        core_series_diff = core_series  
    
    producer_series_random = producer_series_diff.sample(frac=1, random_state=42).reset_index(drop=True)
    core_series_ordered = core_series_diff.reset_index(drop=True)
    min_len = min(len(core_series_ordered), len(producer_series_random))
    core_series_ordered = core_series_ordered.iloc[:min_len]
    producer_series_random = producer_series_random.iloc[:min_len]
    data_random_producer = pd.concat([
        core_series_ordered.rename("Core_Series"),
        producer_series_random.rename("Producer_Series")
    ], axis=1)

    core_series_random = core_series_diff.sample(frac=1, random_state=42).reset_index(drop=True)
    producer_series_ordered = producer_series_diff.reset_index(drop=True)
    min_len = min(len(core_series_random), len(producer_series_ordered))
    producer_series_ordered = producer_series_ordered.iloc[:min_len]
    core_series_random = core_series_random.iloc[:min_len]
    data_random_core = pd.concat([
        core_series_random.rename("Core_Series"),
        producer_series_ordered.rename("Producer_Series")
    ], axis=1)

    try:
        granger_test_results1 = grangercausalitytests(data_random_core[['Core_Series', 'Producer_Series']], maxlag=lags, verbose=True)
        producer_to_core_p_values = []

        for lag in range(1, lags + 1):
            test_result = granger_test_results1[lag][0]['ssr_ftest']
            p_value = test_result[1]
            producer_to_core_p_values.append(p_value)
        
        avg_p_value_producer_to_core = np.mean(producer_to_core_p_values)
    except:
        avg_p_value_producer_to_core = -1
    # elements.append(Paragraph(f"Average p-value for Producer -> Core: {avg_p_value_producer_to_core:.6f}", styles["Normal"])) 
 
   # !---------------------------------------- Core drive producer ----------------------------------------------
    core_series = core_series.shift(k).dropna()
    common_index = core_series.index.intersection(producer_series.index)
    producer_series = producer_series.loc[common_index]
    core_series = core_series.loc[common_index]

    result_core = adfuller(core_series)[1]
    result_producer = adfuller(producer_series)[1]

    if result_core > 0.05 and result_producer > 0.05:
        # --------------Both not stationary---------------------------
        core_series_diff = core_series.diff().dropna()
        producer_series_diff= producer_series.diff().dropna()

    if result_core < 0.05 and result_producer > 0.05:
        # ---------------Producer not stationary-------------------------
        producer_series_diff = producer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_core > 0.05 and result_producer < 0.05:
    # --------------Core not stationary-----------------------
        core_series_diff = core_series.diff().dropna()
        producer_series_diff = producer_series.iloc[1:]

    if result_producer < 0.05 and result_core < 0.05:
        producer_series_diff = producer_series
        core_series_diff = core_series

    try:
        granger_test_results2 = grangercausalitytests(data_random_producer[['Producer_Series', 'Core_Series']], maxlag=lags, verbose=True)
        core_to_producer_p_values = []
        
        for lag in range(1, lags + 1):
            test_result = granger_test_results2[lag][0]['ssr_ftest']
            p_value = test_result[1]
            core_to_producer_p_values.append(p_value)
        avg_p_value_core_to_producer = np.mean(core_to_producer_p_values)
    except:
        avg_p_value_core_to_producer = -1

    # Compute the ratio
    p_value_ratio = avg_p_value_producer_to_core / avg_p_value_core_to_producer
   
    return producer_to_core_p_values, core_to_producer_p_values

def plot_heatmaps_v2(p_value_per_type, k):
    type_keys = list(p_value_per_type.keys())
    
    first_type_key = type_keys[0]
    deltas_in_first_type = list(p_value_per_type[first_type_key].keys())
    
    first_delta = deltas_in_first_type[0]
    communities_in_first_delta = list(p_value_per_type[first_type_key][first_delta].keys())
    
    lag_labels = [str(i) for i in range(1, 8)]
    
    for community in communities_in_first_delta:
        for tkey in type_keys:
            M_producer = np.zeros((7, 7))
            M_core = np.zeros((7, 7))

            for col, delta_str in enumerate(deltas_in_first_type):
                community_dict = p_value_per_type.get(tkey, {}).get(delta_str, {}).get(community, {})
                
                producer_vals = community_dict.get("Producer drive Core", [])
                core_vals = community_dict.get("Core drive Producer", [])
                
                for row in range(7):
                    val_producer = producer_vals[row] if row < len(producer_vals) else np.nan
                    val_core = core_vals[row] if row < len(core_vals) else np.nan
                    M_producer[row, col] = val_producer
                    M_core[row, col] = val_core

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            masked_core = np.where(M_core > 0.05, np.nan, M_core)
            masked_producer = np.where(M_producer > 0.05, np.nan, M_producer)

            # LEFT: 'Producer drive core'
            sns.heatmap(
                masked_producer,
                cmap="YlGnBu",
                annot=M_producer,
                fmt=".5f",
                annot_kws={"size": 8, "color": "black"},
                ax=axes[0],
                xticklabels=deltas_in_first_type,
                yticklabels=lag_labels,
                cbar=True,
                linewidths=0.5,
                linecolor='grey'
            )

            for i in range(M_producer.shape[0]):
                for j in range(M_producer.shape[1]):
                    if M_producer[i, j] > 0.05:
                        axes[0].add_patch(Rectangle((j, i), 1, 1, color='black'))

            axes[0].set_title(f"{community} - {tkey}\nProducer → Core")
            axes[0].set_xlabel("Window Size (Δ)")
            axes[0].set_ylabel("Lag")

            # RIGHT: 'Core drive consumer'
            sns.heatmap(
                masked_core,
                cmap="YlGnBu",
                annot=M_core,
                fmt=".5f",
                annot_kws={"size": 8, "color": "black"},
                ax=axes[1],
                xticklabels=deltas_in_first_type,
                yticklabels=lag_labels,
                cbar=True,
                linewidths=0.5,
                linecolor='grey'
            )
            for i in range(M_core.shape[0]):
                for j in range(M_core.shape[1]):
                    if M_core[i, j] > 0.05:
                        axes[1].add_patch(Rectangle((j, i), 1, 1, color='black'))

            axes[1].set_title(f"{community} - {tkey}\nCore → Producer")
            axes[1].set_xlabel("Window Size (Δ)")
            axes[1].set_ylabel("Lag")

            # Add a supertitle
            plt.suptitle(f"P-value Heatmaps (Lag vs. Δ)\nCommunity: {community}, Type: {tkey}, K:{k}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
 
            # Save figure
            file_name = f"k{k}_{community}_{tkey}_producer_core_heatmap.pdf".replace(" ", "_")
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
                cdc_values = directions_dict["Producer drive Core"]
                if not cdc_sums:
                    cdc_sums = [0.0] * len(cdc_values)
                for i, val in enumerate(cdc_values):
                    cdc_sums[i] += val
                cdc_count += 1

                # 2) "Consumer drive Core"
                c2c_values = directions_dict["Core drive Producer"]
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
                "Producer drive Core": cdc_avgs,
                "Core drive Producer": c2c_avgs
            }
    return result

if __name__ == "__main__":
    # communities = ["astronomy"]
    communities = ["poetry_expanded", "game_development", "climate_change_expanded"]
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
    "science_fiction_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"},}

    types = ["top3"]
    p_value_per_type = {type: {} for type in types}
    old = ["ml_community_1", "new_all_tweets_ml_community_2", "rachel_chess_community"]

    for k in range(1, 10):
        for community in communities:
            if community in old:
                date_range = community_date_ranges[community]
                producers = get_producers(community, "old")
                # producers = get_random_producers(community)
                producer_supply_ts = aggregate_supply_time_series(producers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
                producer_ts = producer_supply_ts

                core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
                core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="old")
            else:
                date_range = community_date_ranges[community]
                producers = get_producers(community, "new")
                # producers = get_random_producers(community)
                producer_supply_ts = aggregate_supply_time_series(producers, community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
                producer_ts = producer_supply_ts

                core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
                core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"], type="new")
            
            core_ts = pd.concat([core_demand_out_ts, core_demand_in_ts])
            core_ts = core_ts.reset_index()
            core_ts = core_ts.groupby('date', as_index=False).sum()
            core_ts = core_ts.set_index('date')

            
            for delta in range(1, 8):
                core_ts = get_ts_window(core_ts, delta)
                producer_ts = get_ts_window(producer_ts, delta)
                for type in types:
                    producer_to_core_p, core_to_producer_p = granger_analysis(core_ts, producer_ts, community, type=type, k=k)
                    if delta not in p_value_per_type[type]:
                        p_value_per_type[type][delta] = {}
                    else:
                        if community not in p_value_per_type[type][delta]:
                            p_value_per_type[type][delta][community] = {}

                    p_value_per_type[type][delta][community] = {
                        "Producer drive Core": producer_to_core_p,
                        "Core drive Producer": core_to_producer_p
                    }

        # result = calculate_avg(p_value_per_type)
        # with open(f"hypothesis/h3/producer_core_values/rank6/game_development_producer_core_{k}.json", "w") as file:
        #     json.dump(p_value_per_type, file)
        plot_heatmaps_v2(p_value_per_type, k)       
    # pdf_file = f"producer_core_analysis_k4.pdf"
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
    #         for relationship in ["Producer drive Core", "Core drive Producer", "P Value Ratio"]:
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

    #     table = Table(table_data)
    #     style = TableStyle([
    #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header row background color
    #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
    #     ('WORDWRAP', (0, 0), (-1, -1)),  
    #     ('FONTSIZE', (0, 0), (-1, -1), 8),
    #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #     ('GRID', (0, 0), (-1, -1), 1, colors.black)
    #     ])

    #     # Alternate row colors for readability
    #     for idx, row in enumerate(table_data[1:], start=1):
    #         bg_color = colors.lightgrey if row[0] % 2 == 0 else colors.whitesmoke
    #         style.add('BACKGROUND', (0, idx), (-1, idx), bg_color)

    #     table.setStyle(style)

    #     elements.append(PageBreak())
    #     elements.append(Paragraph(f"Granger Causality Analysis Results for {type}", styles["Heading2"]))
    #     elements.append(table)

    #     for relationship in ["Producer drive Core", "Core drive Producer", "P Value Ratio"]:
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
        

    