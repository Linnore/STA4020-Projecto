from scipy.stats import norm
import pandas as pd
import talib as ta
#crt_dir = os.path.abspath("")
#data_dir = os.path.abspath("data")
agent_list = [.05, .10, .20, .50, .80, .90]
agent_list2 = [0.3,0.4,0.6,0.7]
#path = './data/Monthly_Excess_Return_Rates.csv'

def data_load(path): 
    '''
    Input: 
    path of the data (Monthly_Excess_Return_Rates.csv)
    Output: df
    '''
    all_data = pd.read_csv(path)
    return all_data

def bayesFormula(returnR, var, a, b):
    n = len(returnR)
    coeff = b**2/(var+n*b**2)
    sigma2 = coeff*var
    miu = (1-n*coeff)*a+coeff*sum(returnR)
    return miu, sigma2

def bayesPredict(stocks, agent_type, batch_size=9, relaxCoeff=8,dynamic=False):
    '''
    This function is used to predict "next-month" return.
    The prior and likelihood are both NORMAL distribution model.
    Prior model has informative parameter settings (based on the stock history)
    its MEAN = Historical mean rate of return; its VARIANCE =  relaxCoeff * Sample VAR
    This is a meaningful setting, if we believe stock market follows a MEAN-REVERTING process. 

    Input: 
    stocks df: monthly excess return from 2010 to 2022
    agent_type: float in (0,1); near 0 -> more conservative for stock return; near 1 -> more aggressive
    batch_size: we use this number of months as Bayes observations
    relaxCoeff: int; bigger -> prediction emphasizes more on the recent batch; smaller -> more stable around historical mean
    Output: predicted df 
    '''
    if dynamic == True:
        return bayesPredict_dynamic(stocks,agent_type = 0.3,batch_size=9, relaxCoeff=8,alpha = 1/3)
    else:
        # Suppose our rolling-horizon prediction starts at:
        starting_date = '2017-01-01'
        # We will use sample variance before this date as "b" in bayes formula.
        date_list = list(stocks['Trdmnt'])
        starting_idx = date_list.index(starting_date)
        stock_list = []

        for name in stocks:
            if name.isdigit():
                stock_list.append(name)
        predict_df = pd.DataFrame(stocks, columns=stock_list)
        timeLine = pd.to_datetime(stocks['Trdmnt'])
        predict_df.index = timeLine

        for name in stock_list:
            returnR = stocks[name]
            hist_R = returnR[:starting_idx]
            B = hist_R.var()
            A = hist_R.mean()
            # For every month after starting date, we use Bayes predict. 
            # We only use "batch_size" monthly data for prediction this step.
            # print('History sample mean:', A)
            for seq_num in range(starting_idx, len(date_list)):
                ref_batch = returnR[seq_num-batch_size:seq_num]
                good = ref_batch[ref_batch==ref_batch]
                var = good.var()
                miu, sigma2 = bayesFormula(returnR=good, var=var, a=A, b=relaxCoeff*B)
                percentile_R = norm.ppf(q=agent_type, loc=miu, scale=(sigma2**0.5))
                if percentile_R < -1 or percentile_R > 1:
                    print('Strange Prediction!')
                predict_df.loc[timeLine[seq_num], name] = percentile_R

                #print(date_list[seq_num], 'Bayes predicted \mu', round(miu,5), \
                #    'Actual return is', round(returnR[seq_num],5), \
                #    'Mean return in last batch', round(good.mean(),5))  
            # print(len(predict_dict[name]), (len(date_list)-starting_idx))

        #print(predict_df.tail())
        #print(stocks.tail())
        return predict_df


def agent_choosing(x):
    x = x[0]
    if x < percentile[0]:
        return agent_list[0]
    elif x < percentile[1]:
        return agent_list[1]
    elif x<percentile[2]:
        return agent_list[2]
    elif x<percentile[3]:
        return agent_list[3]
    #else: return agent_list[4]

def dynamic_agent(index_return,alpha):
    agent_list = [.10, .30, .50, .70, .90]
    index_return = index_return.ewm(alpha = 1/3).mean()
    global percentile
    percentile = []
    for i in agent_list2:
        tmp = index_return.quantile(i).values[0]
        percentile.append(tmp)
    agent = index_return.apply(lambda row: agent_choosing(row.values),axis=1)
    agent.to_csv('agent_distribution.csv')
    return agent


def bayesPredict_dynamic(stocks,agent_type = 0.3,batch_size=9, relaxCoeff=8,alpha = 1/3):
    '''
    This function is used to predict "next-month" return.
    The prior and likelihood are both NORMAL distribution model.
    Prior model has informative parameter settings (based on the stock history)
    its MEAN = Historical mean rate of return; its VARIANCE =  relaxCoeff * Sample VAR
    This is a meaningful setting, if we believe stock market follows a MEAN-REVERTING process. 
    Input: 
    stocks df: monthly excess return from 2010 to 2022
    agent_type: float in (0,1); near 0 -> more conservative for stock return; near 1 -> more aggressive
    batch_size: we use this number of months as Bayes observations
    relaxCoeff: int; bigger -> prediction emphasizes more on the recent batch; smaller -> more stable around historical mean
    Output: predicted df 
    '''
    index_return = pd.read_csv("data/000300.csv",index_col=1,parse_dates=True)
    index_return=index_return.drop(columns=['Indexcd'])
    index_return = index_return.diff(1).fillna(0)
    # Suppose our rolling-horizon prediction starts at:
    starting_date = '2017-01-01'
    # We will use sample variance before this date as "b" in bayes formula.
    date_list = list(stocks['Trdmnt'])
    starting_idx = date_list.index(starting_date)
    stock_list = []
    # agent_type_list = []
    
    

    for name in stocks:
        if name.isdigit():
            stock_list.append(name)
    predict_df = pd.DataFrame(stocks, columns=stock_list)
    timeLine = pd.to_datetime(stocks['Trdmnt'])
    predict_df.index = timeLine
    agent = dynamic_agent(index_return,alpha)
    for name in stock_list:
        returnR = stocks[name]
        hist_R = returnR[:starting_idx]
        B = hist_R.var()
        A = hist_R.mean()
        # For every month after starting date, we use Bayes predict. 
        # We only use "batch_size" monthly data for prediction this step.
        # print('History sample mean:', A)
        for seq_num in range(starting_idx, len(date_list)):
            ref_batch = returnR[seq_num-batch_size:seq_num]
            good = ref_batch[ref_batch==ref_batch]
            var = good.var()
            miu, sigma2 = bayesFormula(returnR=good, var=var, a=A, b=relaxCoeff*B)
            percentile_R = norm.ppf(q=agent[seq_num], loc=miu, scale=(sigma2**0.5))
            if percentile_R < -1 or percentile_R > 1:
                print('Strange Prediction!')
            predict_df.loc[timeLine[seq_num], name] = percentile_R

            #print(date_list[seq_num], 'Bayes predicted \mu', round(miu,5), \
            #    'Actual return is', round(returnR[seq_num],5), \
            #    'Mean return in last batch', round(good.mean(),5))  
        # print(len(predict_dict[name]), (len(date_list)-starting_idx))

    #print(predict_df.tail())
    #print(stocks.tail())
    return predict_df
