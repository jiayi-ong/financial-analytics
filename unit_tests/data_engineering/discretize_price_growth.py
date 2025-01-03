import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from fin_ml.data_engineering.discretize_price_growth import label_intervals, engineer_targets



class TestLabelIntervals(unittest.TestCase):

    def test_1(self):
        # Input data
        pc_arr = np.array([0.21, 0.14, -0.18, 0.55, -0.05])
        intervals = [(-np.inf, -0.20), (-0.20, 0), (0, 0.20), (0.20, np.inf)]

        result = label_intervals(pc_arr, intervals)

        # Expected result
        expected = np.array([0, 1, 1, 1])

        # Assert the result
        assert_array_equal(result, expected)



class TestEngineerTarget(unittest.TestCase):

    def test_one_symbol_full_horizon(self):
        # Input data
        test_weekly = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), 100.00]
            ], 
            columns=['symbol', 'date', 'close']
        )
        test_daily = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-02'), 105.00],
                ['ABC', pd.Timestamp('2025-01-03'), 85.00],
                ['ABC', pd.Timestamp('2025-01-04'), 200] # outside of horizon
            ], 
            columns=['symbol', 'date', 'close']
        )

        intervals = [(-np.inf, -0.20), (-0.20, 0), (0, 0.20), (0.20, np.inf)]

        result = engineer_targets(test_weekly, test_daily, 
                    forecast_horizon=2, horizon_margin=0, intervals=intervals
        )

        # Expected result
        expected = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), np.array([0, 1, 1, 0])]
            ], 
            columns=['symbol', 'week', 'labels']
        )

        # Assert the result
        assert_frame_equal(result, expected)


    def test_one_symbol_within_margin(self):
        # Input data
        test_weekly = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), 100.00]
            ], 
            columns=['symbol', 'date', 'close']
        )
        test_daily = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-02'), 105.00]
            ], 
            columns=['symbol', 'date', 'close']
        )

        intervals = [(-np.inf, -0.20), (-0.20, 0), (0, 0.20), (0.20, np.inf)]

        result = engineer_targets(test_weekly, test_daily, 
                    forecast_horizon=2, horizon_margin=1, intervals=intervals
        )

        # Expected result
        expected = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), np.array([0, 0, 1, 0])]
            ], 
            columns=['symbol', 'week', 'labels']
        )

        # Assert the result
        assert_frame_equal(result, expected)


    def test_one_symbol_exceed_margin(self):
        # Input data
        test_weekly = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), 100.00]
            ], 
            columns=['symbol', 'date', 'close']
        )
        test_daily = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-02'), 105.00]
            ], 
            columns=['symbol', 'date', 'close']
        )

        intervals = [(-np.inf, -0.20), (-0.20, 0), (0, 0.20), (0.20, np.inf)]

        result = engineer_targets(test_weekly, test_daily, 
                    forecast_horizon=2, horizon_margin=0, intervals=intervals
        )

        # Expected result
        expected = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), np.nan]
            ], 
            columns=['symbol', 'week', 'labels']
        )

        # Assert the result
        assert_frame_equal(result, expected)


    def test_one_symbol_no_data(self):
        # Input data
        test_weekly = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), 100.00]
            ], 
            columns=['symbol', 'date', 'close']
        )
        test_daily = pd.DataFrame(
            [], 
            columns=['symbol', 'date', 'close']
        )

        intervals = [(-np.inf, -0.20), (-0.20, 0), (0, 0.20), (0.20, np.inf)]

        result = engineer_targets(test_weekly, test_daily, 
                    forecast_horizon=2, horizon_margin=0, intervals=intervals
        )

        # Expected result
        expected = pd.DataFrame(
            [
                ['ABC', pd.Timestamp('2025-01-01'), np.nan]
            ], 
            columns=['symbol', 'week', 'labels']
        )

        # Assert the result
        assert_frame_equal(result, expected)



if __name__ == '__main__':
    unittest.main()