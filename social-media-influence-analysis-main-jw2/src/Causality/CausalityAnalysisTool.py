import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, \
    InfeasibleTestError
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Sequence, Tuple

import warnings
from TS.TimeSeriesBuilderBase import TimeSeriesBuilderBase
from User.UserType import UserType

warnings.simplefilter(action='ignore', category=FutureWarning)


def gc_score_for_lag(indep_series: Sequence, dep_series: Sequence, lag: int,
                     test: str = "ssr_chi2test", verbose: bool = False) \
        -> float:
    """Return the p value for Granger causality test for <indep_series>
     and <dep_series> with <lag>.
     """
    if lag > 0:
        granger_test = grangercausalitytests(pd.DataFrame({
            "y": dep_series,
            "x": indep_series
        }), maxlag=[lag], verbose=verbose)
        return granger_test[lag][0][test][1]
    elif lag < 0:
        granger_test = grangercausalitytests(pd.DataFrame({
            "y": indep_series,
            "x": dep_series
        }), maxlag=[-lag], verbose=verbose)
        return granger_test[-lag][0][test][1]
    else:
        return 1


def gc_score_for_lags(indep_series: Sequence, dep_series: Sequence,
                      lags: List[int],
                      test: str = "ssr_chi2test", verbose: bool = False) \
        -> List[float]:
    """Return a list of p values for Granger causality test for <indep_series>
    and <dep_series> with <lag>.
    """
    result = []
    for lag in lags:
        try:
            p_val = gc_score_for_lag(indep_series, dep_series, lag, test,
                                     verbose)
        except InfeasibleTestError:
            p_val = 1
        result.append(p_val)
    return result


def is_stationary(x: Sequence, sig_level: float = 0.05) -> bool:
    test_result = adfuller(x)
    p_val = test_result[1]
    return p_val < sig_level


def cos_similarity(seq1: Sequence, seq2: Sequence) -> float:
    """Return the cosine similarity between two sequence.
    """
    # Convert sequences to numpy arrays
    vec1 = np.array(seq1)
    vec2 = np.array(seq2)

    # Reshape arrays to be 2-dimensional
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0, 0]

    return similarity


def cs_for_lag(indep_series: Sequence, dep_series: Sequence, lag: int) -> float:
    assert len(indep_series) == len(dep_series)
    if lag > 0:
        return cos_similarity(indep_series[:-lag], dep_series[lag:])
    elif lag < 0:
        return cos_similarity(indep_series[-lag:], dep_series[:lag])
    else:
        return cos_similarity(indep_series, dep_series)


def cs_for_lags(indep_series: Sequence, dep_series: Sequence,
                lags: List[int]) -> List[float]:
    """
    Precondition: min(lags) < 0 and max(lags) > 0
    """
    min_lag = min(lags)
    max_lag = max(lags)
    indep_slice = indep_series[-min_lag:-max_lag]

    result = []
    for lag in lags:
        if max_lag == lag:
            dep_slice = dep_series[- min_lag + lag:]
        else:
            dep_slice = dep_series[- min_lag + lag: - max_lag + lag]
        assert len(dep_slice) == len(indep_slice)
        result.append(cos_similarity(indep_slice, dep_slice))
    return result


def ols_slope(indep_series: Sequence, dep_series: Sequence) -> \
        Tuple[float, Tuple[float, float]]:
    """Return p-value, lower bound, and upper bound of confidence interval with
    significant level = 0.05.
    """
    # independent variable preprocessing
    indep_series = sm.add_constant(indep_series)

    # model fit
    model = sm.OLS(dep_series, indep_series).fit()
    slope = model.params[1]
    conf_int = tuple(model.conf_int(alpha=0.05)[1])
    return slope, conf_int


def lr_for_lag(indep_series: Sequence, dep_series: Sequence, lag: int) -> \
        Tuple[float, Tuple[float, float]]:
    """Return p-value, lower bound, and upper bound of confidence interval.
    """
    if _list_depth(indep_series) == 1:
        if lag > 0:
            return ols_slope(np.array([indep_series[:-lag], dep_series[:-lag]]).T, dep_series[lag:])
        elif lag < 0:
            return ols_slope(np.array([dep_series[:lag], indep_series[:lag]]).T, indep_series[-lag:])
        else:
            return 0, (0, 0)
    elif _list_depth(indep_series) == 2:
        indep_input = []
        dep_indep = []
        dep_input = []
        if lag > 0:
            # build input list
            for indep_list in indep_series:
                indep_input.extend(indep_list[:-lag])
            for dep_list in dep_series:
                dep_indep.extend(dep_list[:-lag])
                dep_input.extend(dep_list[lag:])

            return ols_slope(np.array([indep_input, dep_indep]).T, dep_input)
        elif lag < 0:
            # build input list
            for indep_list in indep_series:
                indep_input.extend(indep_list[-lag:])
                dep_indep.extend(indep_list[:lag])
            for dep_list in dep_series:
                dep_input.extend(dep_list[:lag])

            return ols_slope(dep_input, indep_input)
        else:
            return 0, (0, 0)


def lr_for_lags(indep_series: Sequence, dep_series: Sequence,
                lags: List[int]) -> Tuple[List[float], List[Tuple[float, float]]]:
    slope_list = []
    conf_int_list = []
    for lag in lags:
        slope, conf_int = lr_for_lag(indep_series, dep_series, lag)
        slope_list.append(slope)
        conf_int_list.append(conf_int)
    return slope_list, conf_int_list


def _list_depth(lst: Sequence) -> int:
    """Return the depth of lst. Assume the input list has at least one element,
    and the depth is uniform across all sublist.
    """
    sub = lst[0]
    if isinstance(sub, list):
        return _list_depth(sub) + 1
    return 1


def ols_for_bins(ts: TimeSeriesBuilderBase, bins: List, lag: int, aggregate: bool, core_node):
    # Create suitable dataframe
    # Our dataframe will have 4 columns: demand, bin_number, time_window, and supply
    df = pd.DataFrame()
    demand = np.array([])
    supply = np.array([])
    bin_number = np.array([])
    time_window = np.array([])
    k = len(ts.time_stamps) - 1
    filename = ''
    for content_type in bins:
        if filename == '':
            filename += str(content_type)
        else:
            filename += '_' + str(content_type)
        consumer_demand = ts.create_time_series(UserType.CONSUMER, content_type, "demand_in_community")
        core_node_demand = ts.create_time_series(UserType.CORE_NODE, content_type, "demand_in_community")
        core_node_supply = ts.create_time_series(UserType.CORE_NODE, content_type, "supply")
        producer_supply = ts.create_time_series(UserType.PRODUCER, content_type, "supply")
        if aggregate:
            demand = np.concatenate((demand, np.add(consumer_demand, core_node_demand)))
            supply = np.concatenate((supply, np.add(producer_supply, core_node_supply)))
        else:
            if core_node is not None:
                demand = np.concatenate((demand, consumer_demand))
                core_node_supply_one = ts.create_time_series(core_node, content_type, "supply")
                supply = np.concatenate((supply, core_node_supply_one))
            else:
                ## for consumer and core nodes
                # demand = np.concatenate((demand, consumer_demand))
                # supply = np.concatenate((supply, core_node_supply))
                ## aggregate core node supply with demand and supply together
                # supply = np.concatenate((supply, np.add(core_node_demand, core_node_supply)))

                ## for core nodes and producer
                demand = np.concatenate((demand, core_node_demand))
                # demand = np.concatenate((supply, np.add(core_node_demand, core_node_supply)))
                supply = np.concatenate((supply, producer_supply))

        bin_number = np.concatenate((bin_number, np.array([content_type] * k)))
        time_window = np.concatenate((time_window, np.array([i + 1 for i in range(k)])))
    df['demand'] = demand
    df['supply'] = supply
    df['bin'] = bin_number
    df['time_window'] = time_window

    # create lagged values for the bin
    for lag in range(1, lag + 1):
        df[f'demand_lag_{lag}'] = df.groupby('bin')['demand'].shift(lag)
        df[f'supply_lag_{lag}'] = df.groupby('bin')['supply'].shift(lag)

    df = df.dropna()
    if core_node is None:
        if aggregate:
            try:
                os.makedirs(f'Data/{filename}')
            except OSError as error:
                print(error)
            df.to_csv(f'Data/{filename}/{filename}.csv', index=False)
        else:
            try:
                os.makedirs(f'Data/{filename}_cns')
            except OSError as error:
                print(error)
            df.to_csv(f'Data/{filename}_cns/{filename}_cns.csv', index=False)

    else:
        try:
            os.makedirs(f'Data/{filename}_one_chess')
        except OSError as error:
            print(error)
        df.to_csv(f'Data/{filename}_one_chess/{filename}_one.csv', index=False)


def ols_for_word(ts: TimeSeriesBuilderBase, lag: int):
    # Create suitable dataframe
    # Our dataframe will have 4 columns: demand, bin_number, time_window, and supply
    df = pd.DataFrame()
    consumer_demand = ts.create_time_series(UserType.CONSUMER, 1, "demand_in_community")
    core_node_demand = ts.create_time_series(UserType.CORE_NODE, 1, "demand_in_community")
    core_node_supply = ts.create_time_series(UserType.CORE_NODE, 1, "supply")
    producer_supply = ts.create_time_series(UserType.PRODUCER, 1, "supply")
    time_window = np.array([i for i in range(len(consumer_demand))])

    df['consumer_demand'] = consumer_demand
    df['producer_supply'] = producer_supply
    df['core_node_demand'] = core_node_demand
    df['core_node_supply'] = core_node_supply
    df['time_window'] = time_window

    # create lagged values for the bin
    for lag in range(1, lag + 1):
        df[f'consumer_demand_lag_{lag}'] = df['consumer_demand'].shift(lag)
        df[f'producer_supply_lag_{lag}'] = df['producer_supply'].shift(lag)
        df[f'core_node_demand_{lag}'] = df['core_node_demand'].shift(lag)
        df[f'core_node_supply_{lag}'] = df['core_node_supply'].shift(lag)

    df = df.dropna()

    try:
        os.makedirs(f'Data/ml_1_single_core_node')
    except OSError as error:
        print(error)
    df.to_csv(f'Data/ml_1_single_core_node/one.csv', index=False)








