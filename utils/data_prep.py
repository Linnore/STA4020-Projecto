import numpy as np
import pandas as pd


def get_excess_return_rates(R_df: pd.DataFrame, rf=None):
    if rf is None:
        rf = pd.read_csv("../data./Monthly_rf_Rate.csv",
                         index_col=0, parse_dates=True)
    R_excess_df = R_df - rf.values
    return R_excess_df
