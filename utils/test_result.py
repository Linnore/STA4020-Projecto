import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_result(R_excess_df,rf_df = 0):
    #净值
    R_excess_df['net_worth'] = 1*(1+R_excess_df['return']).cumprod(axis = 0)
    #累计收益率
    accu_return = R_excess_df['net_worth'].iloc[-1]/1-1
    #年化收益率 一年按365算
    annual_return = (1+accu_return)**(12/len(R_excess_df))-1
    #年化波动率 一年按365算
    annual_vol = R_excess_df['net_worth'].std(ddof = 1)*np.sqrt(12)
    #夏普，rf = 0.01
    rf_df = 0.01
    sharpe_ratio = (annual_return-rf_df)/(annual_vol)
    #最大回撤
    max_dd = ((R_excess_df['net_worth']-R_excess_df['net_worth'].cummax())/R_excess_df['net_worth'].cummax()).min()
    #胜率
    wining_count = len(R_excess_df[R_excess_df['return'] > 0])/len(R_excess_df)
    #karmar ratio 
    karmar = abs(accu_return/max_dd)
    result_df = pd.DataFrame()
    result_df['accu_return'] = [accu_return]
    result_df['annual_return'] = [annual_return]
    result_df['annual_vol'] = [annual_vol]
    result_df['sharpe_ratio'] = [sharpe_ratio]
    result_df['max_dd'] =[max_dd]
    result_df['winning_rate'] = [wining_count]
    result_df['karmar'] = [karmar]
    return result_df