from typing import List, Tuple

import pandas as pd
import numpy as np



def label_intervals(
    price_changes: np.array, intervals: List[Tuple]
) -> np.array:
    """Returns an array of boolean values, each indicating whether 
    a price change within an interval is detected.
    Arguments:
        price_changes (np.array): array of price changes
        intervals (List[Tuple]): each tuple contains the lower and upper boundaries of the intervals
    """
    labels = np.zeros(len(intervals), dtype=int)
    
    for i, (low, upp) in enumerate(intervals):
        if np.any((price_changes > low) & (price_changes <= upp)):
            labels[i] = 1
            
    return labels



def engineer_targets(
    weekly_df: pd.DataFrame, daily_df: pd.DataFrame,
    forecast_horizon: int, horizon_margin: int, intervals: List[Tuple]
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
    Prerequisites:
        1. weekly_df and daily_df must be sorted by symbol and date (ascending)
        2. the 'date' columns of weekly_df and daily_df must be formated as datetime
    """
    # Initialize a list to store results
    results = []

    # Process each stock symbol separately
    for symbol, group in weekly_df.groupby('symbol'):

        # Iterate through each week's closing price
        for i, row in group.iterrows():
            current_date = row['date']
            current_price = row['close']

            # Define the forecast horizon
            horizon_end_date = current_date + pd.DateOffset(days=forecast_horizon)
            
            # Get data for the symbol
            symbol_df = daily_df.query("symbol == @symbol")
            
            # Check for days missing from the forecast horizon
            if symbol_df.shape[0] == 0 or (horizon_end_date - symbol_df['date'].max()).days > horizon_margin:
                # Append results
                results.append({'symbol': symbol, 'week': current_date, 'labels': np.nan})
                continue

            # Get daily prices within the forecast horizon
            future_prices = (
                symbol_df.query("(date > @current_date) & (date <= @horizon_end_date)")
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