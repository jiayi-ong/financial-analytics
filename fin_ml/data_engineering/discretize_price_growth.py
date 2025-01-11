from typing import List, Tuple

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm



def has_consecutive_ones(arr: np.array, N: int) -> bool:
    """Returns True iff the given array has N consecutive 1s.
    Examples:
        >>> import numpy as np
        >>> has_consecutive_ones(np.array([0, 1, 1, 1, 0, 1, 1]), 3)
        True
        >>> has_consecutive_ones(np.array([0, 1, 0, 1, 1, 0, 1]), 2)
        True
        >>> has_consecutive_ones(np.array([False, True, False, True, False]), 1)
        True
        >>> has_consecutive_ones(np.array([0, 0, 0, 0, 0, 0]), 1)
        False
        >>> has_consecutive_ones(np.array([1, 1, 1, 1, 1]), 6)
        False
    """
    count = 0  # Counter for consecutive 1s
    for num in arr:
        if num == 1:
            count += 1  # Increment counter if the element is 1
            if count == N:  # If counter reaches N, return True
                return True
        else:
            count = 0  # Reset counter if the element is 0
    return False



def generate_symmetric_intervals(boundaries: List[float]) -> List[Tuple]:
    """Returns a list of symmetric intervals given the one-sided boundaries input.
    NOTE: must be symmetric about 0 (boundaries must include 0 as the first item).
    Arguments:
        boundaries (List[float]): the boundary values on the positive side.
    Returns:
        List[Tuple]: each item contains the starting and ending value of the interval.
    Examples:
        >>> generate_symmetric_intervals([0, 1, 2])
        [(-2, -1), (-1, 0), (0, 1), (1, 2)]
    """
    # define one-sided intervals
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # replicate other side with symmetry
    intervals += [(-e[1],-e[0]) for e in intervals]
    intervals = sorted(intervals)

    return intervals



def label_intervals(
    price_changes: np.array, intervals: List[Tuple], persistence: int=1
) -> np.array:
    """Returns an array of boolean values, each indicating whether 
    a price change within an interval is detected.
    Arguments:
        price_changes (np.array): array of price changes relative to a single reference day
        intervals (List[Tuple]): each tuple contains the lower and upper boundaries of the intervals
        persistence (int): the required number of consecutive trading days for a price change to be considered.
            e.g. if persistence=2 but a 20% growth is only sustained for 1 day, it does not count.
    Examples:
        >>> import numpy as np
        >>> price_changes = np.array([0.01, -0.2, -0.22, -0.1, 0.25, 0.26])
        >>> intervals = [(-0.3,-0.15),(-0.15,0),(0,0.15),(0.15,0.3)]
        >>> label_intervals(price_changes, intervals, persistence=2)
        array([1, 0, 0, 1])
        >>> label_intervals(price_changes, intervals, persistence=1)
        array([1, 1, 1, 1])
    """
    labels = np.zeros(len(intervals), dtype=int)
    
    for i, (low, upp) in enumerate(intervals):
        bool_array = (price_changes > low) & (price_changes <= upp)
        if has_consecutive_ones(bool_array, persistence):
            labels[i] = 1
            
    return labels



def engineer_targets(
    weekly_df: pd.DataFrame, daily_df: pd.DataFrame,
    forecast_horizon: int, horizon_margin: int, intervals: List[Tuple],
    persistence: int=1
) -> pd.DataFrame:
    """Returns a DataFrame that maps each stock's weekly data to the multi-label target variable,
    which indicates the discrete levels of price changes actualized within the forecast horizon relative to
    the week's closing price. NOTE: if there is not enough daily data to fill the forecast horizon, (i.e.
    not enough data to conclude that the growth levels will be be observed), NA will be returned as the target value.
    Arguments:
        weekly_df (pd.DataFrame): contains the weekly closing stock prices of each symbol
        daily_df (pd.DataFrame): contains the daily closing stock prices of each symbol
        forecast_horizon (int): number of days in the forecast horizon
        horizon_margin (int): the allowance for the number of days missing from the forecast horizon
            (horizon_margin < forecast_horizon)
        intervals: each tuple contains the lower and upper boundaries of the price change intervals
        persistence (int): the required number of consecutive trading days for a price change to be considered.
    Prerequisites:
        1. weekly_df and daily_df must be sorted by symbol and date (ascending)
        2. the 'date' columns of weekly_df and daily_df must be formated as datetime
    """
    # Initialize a list to store results
    results = []

    # Process each stock symbol separately
    for symbol, w_data in weekly_df.groupby('symbol'):
            
        # Get daily data for the symbol
        d_data = daily_df.query("symbol == @symbol")
        if d_data.shape[0] == 0:
            warnings.warn(f"No daily data found for symbol '{symbol}', continuing to next symbol.")
            continue

        # Define the forecast horizon
        w_data['horizon_end_date'] = w_data['date'] + pd.DateOffset(days=forecast_horizon)

        # Days missing from each week's forecast horizon 
        # = horizon end date - maximum date of daily data
        max_daily_date = d_data['date'].max()
        w_data['days_missing_from_horizon'] = (w_data['horizon_end_date'] - max_daily_date).dt.days

        # Iterate through each week's closing price
        for i, row in tqdm(w_data.iterrows(), desc=f"Calculating for {symbol}", total=w_data.shape[0]):
            current_date = row['date']
            current_price = row['close']
            horizon_end_date = row['horizon_end_date']
            days_missing_from_horizon = row['days_missing_from_horizon']
            
            # Check for days missing from the forecast horizon
            if days_missing_from_horizon > horizon_margin:
                # Append results
                results.append({'symbol': symbol, 'week': current_date, 'labels': np.nan})
                continue

            # Get daily closing prices within the forecast horizon
            future_prices = (
                d_data.query("(date > @current_date) & (date <= @horizon_end_date)")
                ['close']
                .values
            )

            # Compute price growths
            price_changes = (future_prices - current_price) / current_price

            # Label based on intervals
            labels = label_intervals(price_changes, intervals)

            # Append results
            results.append({'symbol': symbol, 'week': current_date, 'labels': labels})

    # Convert results to a DataFrame
    target_df = pd.DataFrame(results)

    return target_df



if __name__ == "__main__":
    import doctest
    doctest.testmod()