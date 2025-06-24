import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")

# first get the core user and core users' followers
def get_core_consumers(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    #core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})

    consumers = [x for x in core_user_item["local follower list"]]
    return consumers

def get_core_consumers_restricted(db: str):
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    #user_df['userid'] = user_df['userid'].astype(int)

    core_user_item = user_collection.find_one({"rank": 0})
    #core_user_item = user_collection.find_one({"userid": 228660231})
    core_user = core_user_item["userid"]

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})

    consumers = core_user_item["local follower list"]
    consumer_df = user_df[user_df['userid'].isin(consumers)]
    consumer_df = consumer_df[consumer_df['local follower list'].apply(lambda x: core_user in x)]

    consumers = consumer_df['userid'].to_list()
    return consumers

def aggregate_demand_in_time_series(consumers, db, start_date=None, end_date=None):
    time_series = MONGO_CLIENT[db]['retweet_in_time_series_word']
    time_series_df = pd.DataFrame(list(time_series.find()))
    #time_series_df['user_id'] = time_series_df['user_id'].astype(int)

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

    if start_date and end_date:
        filtered_topics_df = topics_df.loc[start_date:end_date]
        return filtered_topics_df
    return topics_df

def aggregate_demand_out_time_series(consumers, db, start_date=None, end_date=None):
    time_series = MONGO_CLIENT[db]['retweet_out_time_series_word']
    time_series_df = pd.DataFrame(list(time_series.find()))
    #time_series_df['user_id'] = time_series_df['user_id'].astype(int)

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
    if start_date and end_date:
        filtered_topics_df = topics_df.loc[start_date:end_date]
        return filtered_topics_df
    return topics_df

def get_core_supply_time_series(db, start_date=None, end_date=None):
    user_collection = MONGO_CLIENT[db]['user_info']
    #core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['og_tweet_time_series_word']
    og_df = pd.DataFrame(list(og_ts.find()))
    #og_df['user_id'] = og_df['user_id'].astype(int)
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
    if start_date and end_date:
        filtered_topics_df = topics_df.loc[start_date:end_date]
        return filtered_topics_df
    return topics_df

def get_core_demand_in_time_series(db, start_date=None, end_date=None):
    user_collection = MONGO_CLIENT[db]['user_info']
    #core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_in_time_series_word']
    og_df = pd.DataFrame(list(og_ts.find()))
    #og_df['user_id'] = og_df['user_id'].astype(int)
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
    if start_date and end_date:
        filtered_topics_df = topics_df.loc[start_date:end_date]
        return filtered_topics_df
    return topics_df

def get_core_demand_out_time_series(db, start_date=None, end_date=None):
    user_collection = MONGO_CLIENT[db]['user_info']
    #core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_out_time_series_word']
    og_df = pd.DataFrame(list(og_ts.find()))
    #og_df['user_id'] = og_df['user_id'].astype(int)
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
    if start_date and end_date:
        filtered_topics_df = topics_df.loc[start_date:end_date]
        return filtered_topics_df
    return topics_df

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

def granger_analysis(consumer_demand_ts, core_supply_ts, db, type, styles, elements, lags=7):
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

    consumer_topics = list(consumer_demand_sums.index)
    core_topics = list(core_supply_sums.index)

    with open(f"{db}_consumer.json", 'w') as file:
        json.dump(consumer_topics, file, indent=4)
    
    with open(f"{db}_core.json", 'w') as file:
        json.dump(core_topics, file, indent=4)

    exit(0)


    # combines top 5 topics
    # ---------------------------------------------------------
    if type == "top_5_core":
        top_5_core_supply = core_supply_sums.nlargest(5)

        consumer_series = consumer_demand_ts_aligned[top_5_core_supply.index]
        core_series = core_supply_ts_aligned[top_5_core_supply.index]

        consumer_series = consumer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)
    
    if type == "top_5_consumer":
        top_5_consumer_demand = consumer_demand_sums.nlargest(5)

        consumer_series = consumer_demand_ts_aligned[top_5_consumer_demand.index]
        core_series = core_supply_ts_aligned[top_5_consumer_demand.index]

        consumer_series = consumer_series.sum(axis=1)
        core_series = core_series.sum(axis=1)


    consumer_zeros = (consumer_series == 0).sum()
    core_zeros = (core_series == 0).sum()
    elements.append(Paragraph(f"- Zeros Core: {core_zeros} / {len(core_series)}", styles["Normal"]))
    elements.append(Paragraph(f"- Zeros: Consumers: { consumer_zeros} / {len(consumer_series)}", styles["Normal"]))

    result_consumer = adfuller(consumer_series)[1]
    result_core = adfuller(core_series)[1]

    if result_consumer > 0.05 and result_core > 0.05:
        # --------------Both not stationary---------------------------
        elements.append(Paragraph("- Both not stationary", styles["Normal"]))
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff= consumer_series.diff().dropna()

    if result_consumer < 0.05 and result_core > 0.05:
        # ---------------Core not stationary-------------------------
        elements.append(Paragraph("- Core not stationary", styles["Normal"]))
        core_series_diff = core_series.diff().dropna()
        consumer_series_diff = consumer_series.iloc[1:]

    if result_consumer > 0.05 and result_core < 0.05:
    # --------------Consumer not stationary-----------------------
        elements.append(Paragraph("- Consumer not stationary", styles["Normal"]))
        consumer_series_diff = consumer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_consumer < 0.05 and result_core < 0.05:
        consumer_series_diff = consumer_series
        core_series_diff = core_series

    #----------------------------Granger Analysis-----------------------------------------------
    elements.append(Paragraph("Core drive consumer", styles["Heading4"]))
    data = pd.concat([consumer_series_diff, core_series_diff], axis=1)
    data.columns = ['Consumer_Series', 'Core_Series']
    data = data.dropna()
    granger_test_results1 = grangercausalitytests(data[['Consumer_Series', 'Core_Series']], maxlag=lags, verbose=True)

    for lag in range(1, lags + 1):
        test_result = granger_test_results1[lag][0]['ssr_ftest']
        f_statistic = test_result[0]
        p_value = test_result[1]
        elements.append(Paragraph(f"Lag {lag} | F-Statistic: {f_statistic:.6f} | P-value: {p_value:.6f}", styles["Normal"]))
    
    print("---------------------------------------- Consumer drive core ----------------------------------------------")
    elements.append(Paragraph("Consumer drive core", styles["Heading4"]))
    granger_test_results2 = grangercausalitytests(data[['Core_Series', 'Consumer_Series']], maxlag=lags, verbose=True)
    for lag in range(1, lags + 1):
        test_result = granger_test_results2[lag][0]['ssr_ftest']
        f_statistic = test_result[0]
        p_value = test_result[1]
        elements.append(Paragraph(f"Lag {lag} | F-Statistic: {f_statistic:.6f} | P-value: {p_value:.6f}", styles["Normal"]))
   


if __name__ == "__main__":
    # community = "ml_community_1"
    communities =  ["neuroscience_expanded"]
    community_date_ranges = {
    "astronomy": {"start_date": "2023-06-19", "end_date": "2024-07-14"},
    "climate_change": {"start_date": "2023-08-03", "end_date": "2024-06-24"}, 
    "comics": {"start_date": "2023-05-12", "end_date": "2024-08-03"}, 
    "history": {"start_date": "2023-07-30", "end_date": "2024-08-20"},
    "microbiology": {"start_date": "2023-08-17", "end_date": "2024-08-21"}, 
    "neuroscience": {"start_date": "2023-07-24", "end_date": "2024-07-15"}, 
    "poetry": {"start_date": "2023-07-16", "end_date": "2024-08-23"}, 
    "history_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "comics_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "astronomy_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "game_development": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "urban_planning": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "math": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "poetry_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "climate_change_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "neuroscience_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}, 
    "science_fiction_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"},}

    for community in communities:
        # consumers = get_core_consumers_restricted(community)
        consumers = get_core_consumers(community)
        date_range = community_date_ranges[community]
        consumer_demand_in_ts = aggregate_demand_in_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        consumer_demand_out_ts = aggregate_demand_out_time_series(consumers, community, start_date=date_range["start_date"], end_date=date_range["end_date"])

        core_supply_ts = get_core_supply_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"])

        # NOTE: combine core demand and supply
        # ----------------------------------------------------------------
        combined_df = pd.concat([core_supply_ts, core_demand_in_ts])
        combined_df = combined_df.reset_index()
        combined_df = combined_df.groupby('date', as_index=False).sum()

        core_ts = pd.concat([core_demand_out_ts, combined_df])
        core_ts = core_ts.reset_index()
        core_ts = core_ts.groupby('date', as_index=False).sum()
        core_ts = core_ts.set_index('date')
        core_ts = combined_df

        # ----------------------------------------------------------------

        # NOTE: combine consumer demand in and out
        # ----------------------------------------------------------------
        # consumer_ts = pd.concat([consumer_demand_in_ts, consumer_demand_out_ts])
        consumer_ts = consumer_demand_in_ts
        consumer_ts = consumer_ts.reset_index()
        consumer_ts = consumer_ts.groupby('date', as_index=False).sum()
        consumer_ts = consumer_ts.set_index('date')
        # consumer_ts = consumer_demand_in_ts
        # ----------------------------------------------------------------

        types = ["top_5_consumer"]

        # initialize PDF
        pdf_file = f"granger_analysis2_results_{community}_all.pdf"
        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()
        elements = []

        for delta in range(1, 8):
            core_ts = get_ts_window(core_ts, delta)
            consumer_ts = get_ts_window(consumer_ts, delta)
            elements.append(Paragraph(f"Delta: {delta}", styles["Heading2"]))
            for type in types:
                elements.append(Paragraph(f"Topic Type: {type}", styles["Heading3"]))
                result = granger_analysis(consumer_ts, core_ts, community, type=type, styles=styles, elements=elements)
                # exit(0)
        # # build the pdf at the end
        doc.build(elements)

    # NOTE: checking for the longest continuous period
    # -----------------------------------------------------------------------------

    df = consumer_ts.sum(axis=1)
    df_in = core_demand_in_ts.sum(axis=1)
    df_out = core_demand_out_ts.sum(axis=1)

    threshold = 2

    continuous_periods = df.rolling(window=threshold+1, min_periods=1).apply(
        lambda x: np.sum(x == 0) <= threshold).astype(bool)

    continuous_periods_group = (continuous_periods != continuous_periods.shift()).cumsum()

    group_lengths = continuous_periods.groupby(continuous_periods_group).sum()

    longest_group = group_lengths.idxmax()
    longest_length = group_lengths.max()

    longest_period = df[continuous_periods_group == longest_group]

    print(f"Longest Continuous Period Length: {longest_length}")
    print(longest_period.to_string())
    # -----------------------------------------------------------------------------

    