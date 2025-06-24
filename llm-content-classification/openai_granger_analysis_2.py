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
def get_producers(db: str):
    # ---------------------- All community members --------------------------------
    # user_collection = MONGO_CLIENT[db]['user_info']
    # all_users = pd.DataFrame(list(user_collection.find()))
    # all_users['userid'] = all_users['userid'].astype(int)

    # core_user = all_users[all_users['rank'] == 0]['userid']

    # producers = []
    # for index, user in all_users.iterrows():
    #     if user['userid'] not in core_user.values:
    #         producers.append(user['userid'])

    # ----------------------- Producers that follow the core agent -----------------
    user_collection = MONGO_CLIENT[db]['user_info']
    user_df = pd.DataFrame(list(user_collection.find()))
    #user_df['userid'] = user_df['userid'].astype(int)

    # core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})
    core_user = core_user_item["userid"]

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})

    producers = core_user_item["local following list"]
    producer_df = user_df[user_df['userid'].isin(producers)]
    producer_df = producer_df[producer_df['local following list'].apply(lambda x: core_user in x)]

    producers = producer_df['userid'].to_list()
    return producers


def aggregate_demand_in_time_series(producers, db, start_date, end_date):
    time_series = MONGO_CLIENT[db]['retweet_in_time_series_openai']
    time_series_df = pd.DataFrame(list(time_series.find()))
    #time_series_df['user_id'] = time_series_df['user_id'].astype(int)

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

def aggregate_demand_out_time_series(producers, db, start_date, end_date):
    time_series = MONGO_CLIENT[db]['retweet_out_time_series_openai']
    time_series_df = pd.DataFrame(list(time_series.find()))
    #time_series_df['user_id'] = time_series_df['user_id'].astype(int)

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

def aggregate_supply_time_series(producers, db, start_date, end_date):
    time_series = MONGO_CLIENT[db]['og_tweet_time_series_openai']
    time_series_df = pd.DataFrame(list(time_series.find()))
    #time_series_df['user_id'] = time_series_df['user_id'].astype(int)

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

def get_core_demand_in_time_series(db, start_date, end_date):
    user_collection = MONGO_CLIENT[db]['user_info']
    # core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_in_time_series_openai']
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
    filtered_topics_df = topics_df.loc[start_date:end_date]
    
    return filtered_topics_df


def get_core_demand_out_time_series(db, start_date, end_date):
    user_collection = MONGO_CLIENT[db]['user_info']
    # core_user_item = user_collection.find_one({"userid": 228660231})
    core_user_item = user_collection.find_one({"rank": 0})

    if core_user_item is None:
        core_user_item = user_collection.find_one({"rank": 1})
    core_user = core_user_item["userid"]

    og_ts = MONGO_CLIENT[db]['retweet_out_time_series_openai']
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

def granger_analysis(core_demand_ts, producer_supply_ts, db, type, styles, elements, lags=7):
    earliest_date = min(core_demand_ts.index.min(), producer_supply_ts.index.min())
    latest_date = max(core_demand_ts.index.max(), producer_supply_ts.index.max())

    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    
    core_demand_ts_aligned = core_demand_ts.reindex(full_date_range, fill_value=0)
    producer_supply_ts_aligned = producer_supply_ts.reindex(full_date_range, fill_value=0)

    producer_sums = producer_supply_ts_aligned.sum().sort_values(ascending=False)
    core_sums = core_demand_ts_aligned.sum().sort_values(ascending=False)

    # print(producer_sums.sum())
    # print(core_sums.sum())
    # exit(0)

    # read the file that contains the consumer and core topics
    with open(f"{db}/{db}_consumer.json", 'r') as f:
        consumer_topics = json.load(f)
    
    with open(f"{db}/{db}_core.json", 'r') as f:
        core_topics = json.load(f)
    
    # if type == "top_consumer":
    #     for topic in consumer_topics:
    #         if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
    #             producer_series = producer_supply_ts_aligned[topic]
    #             core_series = core_demand_ts_aligned[topic]

    # if type == "top_core":
    #     for topic in core_topics:
    #         if (topic in producer_supply_ts_aligned.columns) and (topic in core_demand_ts_aligned.columns):
    #             producer_series = producer_supply_ts_aligned[topic]
    #             core_series = core_demand_ts_aligned[topic]
    
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
    
    # if type == "all":
    #     producer_series = producer_supply_ts_aligned.sum(axis=1)
    #     core_series = core_demand_ts_aligned.sum(axis=1)


    core_zeros = (core_series == 0).sum()
    producer_zeros = (producer_series == 0).sum()
    elements.append(Paragraph(f"- Zeros Core: {core_zeros} / {len(core_series)}", styles["Normal"]))
    elements.append(Paragraph(f"- Zeros: Producers: {producer_zeros} / {len(producer_series)}", styles["Normal"]))

    result_core = adfuller(core_series)[1]
    result_producer = adfuller(producer_series)[1]

    if result_core > 0.05 and result_producer > 0.05:
        # --------------Both not stationary---------------------------
        elements.append(Paragraph("- Both not stationary", styles["Normal"]))
        core_series_diff = core_series.diff().dropna()
        producer_series_diff= producer_series.diff().dropna()

    if result_core < 0.05 and result_producer > 0.05:
        # ---------------Producer not stationary-------------------------
        elements.append(Paragraph("- Producer not stationary", styles["Normal"]))
        producer_series_diff = producer_series.diff().dropna()
        core_series_diff = core_series.iloc[1:]

    if result_core > 0.05 and result_producer < 0.05:
    # --------------Core not stationary-----------------------
        elements.append(Paragraph("- Core not stationary", styles["Normal"]))
        core_series_diff = core_series.diff().dropna()
        producer_series_diff = producer_series.iloc[1:]

    if result_producer < 0.05 and result_core < 0.05:
        producer_series_diff = producer_series
        core_series_diff = core_series

    #----------------------------Granger Analysis-----------------------------------------------
    elements.append(Paragraph("Producer drive core", styles["Heading4"]))
    data = pd.concat([core_series_diff, producer_series_diff], axis=1)
    data.columns = ['Core_Series', 'Producer_Series']
    data = data.dropna()
    granger_test_results1 = grangercausalitytests(data[['Core_Series', 'Producer_Series']], maxlag=lags, verbose=True)

    for lag in range(1, lags + 1):
        test_result = granger_test_results1[lag][0]['ssr_ftest']
        f_statistic = test_result[0]
        p_value = test_result[1]
        elements.append(Paragraph(f"Lag {lag} | F-Statistic: {f_statistic:.6f} | P-value: {p_value:.6f}", styles["Normal"]))
    
    print("---------------------------------------- Core drive producer ----------------------------------------------")
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Core drive producer", styles["Heading4"]))
    
    granger_test_results2 = grangercausalitytests(data[['Producer_Series', 'Core_Series']], maxlag=lags, verbose=True)
    for lag in range(1, lags + 1):
        test_result = granger_test_results2[lag][0]['ssr_ftest']
        f_statistic = test_result[0]
        p_value = test_result[1]
        elements.append(Paragraph(f"Lag {lag} | F-Statistic: {f_statistic:.6f} | P-value: {p_value:.6f}", styles["Normal"]))

if __name__ == "__main__":
    # community = "rachel_chess_community"
    communities = ["history_expanded"]
    community_date_ranges = {
    "astronomy": {"start_date": "2023-06-19", "end_date": "2024-07-14"},
    "climate_change": {"start_date": "2023-08-03", "end_date": "2024-06-24"}, 
    "comics": {"start_date": "2023-05-12", "end_date": "2024-08-03"}, 
    "history": {"start_date": "2023-07-30", "end_date": "2024-08-20"},
    "microbiology": {"start_date": "2023-08-17", "end_date": "2024-08-21"}, 
    "neuroscience": {"start_date": "2023-07-24", "end_date": "2024-07-15"}, 
    "poetry": {"start_date": "2023-07-16", "end_date": "2024-08-23"}, 
    "history_expanded": {"start_date": "2024-02-25", "end_date": "2025-02-26"}}

    for community in communities:
        date_range = community_date_ranges[community]
        producers = get_producers(community)
        producer_supply_ts = aggregate_supply_time_series(producers, community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        producer_ts = producer_supply_ts

        core_demand_in_ts = get_core_demand_in_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        core_demand_out_ts = get_core_demand_out_time_series(community, start_date=date_range["start_date"], end_date=date_range["end_date"])
        core_ts = pd.concat([core_demand_out_ts, core_demand_in_ts])
        core_ts = core_ts.reset_index()
        core_ts = core_ts.groupby('date', as_index=False).sum()
        core_ts = core_ts.set_index('date')
        # core_ts = core_demand_out_ts

        types = ["top_5_core", "top_5_consumer"]

        # initialize PDF
        pdf_file = f"granger_analysis_results_{community}_all.pdf"
        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()
        elements = []

        for delta in range(1, 8):
            core_ts = get_ts_window(core_ts, delta)
            producer_ts = get_ts_window(producer_ts, delta)
            elements.append(Paragraph(f"Delta: {delta}", styles["Heading2"]))
            for type in types:
                elements.append(Paragraph(f"Topic Type: {type}", styles["Heading3"]))
                result = granger_analysis(core_ts, producer_ts, community, type=type, styles=styles, elements=elements)
        # build the pdf at the end
        doc.build(elements)


    # NOTE: checking for the longest continuous period
    # -----------------------------------------------------------------------------
    # df = core_ts.sum(axis=1)

    # threshold = 2

    # continuous_periods = df.rolling(window=threshold+1, min_periods=1).apply(
    #     lambda x: np.sum(x == 0) <= threshold).astype(bool)

    # continuous_periods_group = (continuous_periods != continuous_periods.shift()).cumsum()

    # group_lengths = continuous_periods.groupby(continuous_periods_group).sum()

    # longest_group = group_lengths.idxmax()
    # longest_length = group_lengths.max()

    # longest_period = df[continuous_periods_group == longest_group]

    # print(f"Longest Continuous Period Length: {longest_length}")
    # print(longest_period.to_string())
    # -----------------------------------------------------------------------------

    