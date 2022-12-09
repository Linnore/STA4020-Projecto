import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_result(df,rf = 0.03):
    #净值
    df['net_worth'] = 1*(1+df['return']).cumprod(axis = 0)
    #累计收益率
    accu_return = df['net_worth'].iloc[-1]/1-1
    #年化收益率 一年按365算
    annual_return = (1+accu_return)**(12/len(df))-1
    #年化波动率 一年按365算
    annual_vol = df['net_worth'].std(ddof = 1)*np.sqrt(12)
    #夏普，rf = 0.01
    rf = 0.01
    sharpe_ratio = (annual_return-rf)/(annual_vol)
    #最大回撤
    max_dd = ((df['net_worth']-df['net_worth'].cummax())/df['net_worth'].cummax()).min()
    #胜率
    wining_count = len(df[df['return'] > 0])/len(df)
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