from sympy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import time

stock_list = [600519,600030,601857,601628]
def data_load(stock_id):
    ID_str = str(stock_id)
    all_data = pd.read_csv('./clean_data/cleaned_data.csv')
    a_stock = pd.DataFrame(all_data, columns=['date', ID_str])
    monthR_list = list(a_stock[ID_str])
    date_list = list(a_stock['date'])
    return a_stock, date_list, monthR_list  # Return a list of monthly return of stock id

def BayesPeak(returnR, sample_var, model='normal', batch_size=1):
    X = symbols('X')
    miu = symbols('miu')
    var1 = 0.01
    n = 2 # Degree of freedom using in t-distribution prior model 
    prior_model_dict = {'normal':1/((2*pi*var1)**0.5)*exp(-miu**2/(2*var1)), 'cauchy':1/(pi*(1+miu**2)), 't':float(gamma((n+1)/2))/((n*pi)**0.5*float(gamma(n/2)))*(1+miu**2/n)**(-(1+n)/2)}
    targ_model_dict = {'normal': 1/((2*pi*sample_var)**0.5)*exp(-(X-miu)**2/(2*sample_var))}
    paraPr = prior_model_dict[model] # Distribution of parameter miu (default assume Normal)
    targPr = targ_model_dict['normal'] # Pdf of X (assume Normal with mean miu)
    postList = [paraPr]
    minimumR = min(returnR)
    maximumR = max(returnR)
    for idx in range(0,len(returnR),batch_size):
        likelihood = 1
        for i in range(batch_size):
            observe_data = returnR[idx+i]
            likelihood = targPr.subs(X, observe_data)*likelihood
            if idx+i >= len(returnR)-1:
                break
        prior = postList[-1]
        if model == 'normal':
            normalizingConstant = float(integrate(likelihood*prior,(miu, -oo, oo))) 
            # This is the exact normalizing constant for normal prior.
        else:
            normalizingConstant = 10**72  # float(integrate(likelihood*prior,(miu, -10, 10))) 
        posterior = likelihood*prior / normalizingConstant
        posterior = simplify(posterior)
        postList.append(posterior)
        print('Iteration', idx, 'is done: posterior', posterior)
    print(postList,'the lenth of postList:',len(postList))

    final_belief = postList[-1]
    # x_axis = [0.01*i for i in range(-100,100)]
    # last_pdf = [final_belief.subs(miu, i) for i in x_axis]
    # prior_pdf = [paraPr.subs(miu,i) for i in x_axis]
    # plt.plot(x_axis,last_pdf)
    # plt.plot(x_axis,prior_pdf)
    # for post in postList[::10]:  # see the evolution of posterior distribution 
    #     post_pdf = [post.subs(miu,i) for i in x_axis]
    #     plt.plot(x_axis, post_pdf)
    # plt.show()
    if model == 'normal':
        miu_bayes = solve(Eq(log(final_belief).diff(miu), 0), miu)[0] #(lhs: Any, rhs: Any) # print(miu_bayes)# is a list.
    else:
        miu_bayes = solveMax(final_belief, miu, minimumR, maximumR)
    return miu_bayes

def solveMax(func, var, lowerbdd, upperbdd, acc=6):
    derivative = func.diff(var)
    p1, p2 = lowerbdd, upperbdd
    while p2-p1 > 10**(-acc):
        mid = (p1+p2)/2
        midfunc = derivative.subs(var, mid)
        p1func = derivative.subs(var, p1)
        p2func = derivative.subs(var, p2)
        if midfunc*p1func<0:
            p2 = mid
        elif midfunc*p2func<0:
            p1 = mid
        elif midfunc == 0:
            return mid
        else:
            print('Bisection error!')
    print('The worst and best daily returns are', lowerbdd, upperbdd, 'respectively, I use bisection to find the peak of posterior miu=', mid)
    return (p1+p2)/2

def BayesPredict(stockID, history=10, acc=6):
    a_stock, date_list, returnR = data_load(stockID)
    R_np = np.array(returnR)
    sample_var = np.var(R_np)
    predicted_df = a_stock.copy()
    for idx in range(1, len(date_list)):
        if idx < history:
            bayes_mean_estimate = BayesPeak(returnR[:idx], sample_var=sample_var, model='normal', batch_size=2)
        else:
            bayes_mean_estimate = BayesPeak(returnR[idx-history:idx], sample_var=sample_var, model='normal', batch_size=2)
            # (model = 't', 'cauchy')

        print('Bayes estimate the mean return of month', date_list[idx], 'is', bayes_mean_estimate)
        print('The true monthly return is', returnR[idx])
            
    predicted_df.loc[idx, str(stockID)] = bayes_mean_estimate
    predicted_df.to_csv(str(stockID)+'BayesPredict.csv')
    predicted_list = list(predicted_df[str(stockID)])
    return predicted_df,  predicted_list
    

def getAllmean():
    mean_var_list = []
    mean_list = []
    for i in range(len(stock_list)):
        returnR = data_load(stock_id=i)
        R_np = np.array(returnR)
        sample_var = np.var(R_np)
        sample_mean = np.mean(R_np)
        mean_var_list.append([sample_mean, sample_var])
        mean_list.append(sample_mean)
    print('All the stocks are listed here:', stock_list)
    df = pd.DataFrame(data=mean_var_list, index=stock_list, columns=['2010-2018 sample mean (RV: daily log-return)', '2001-2018 sample variance (RV: daily log-return)'])
    df.to_csv('Mean-var of daily log-return.csv')
    print('Successfully save in csv.')
    max_return = max(mean_list)
    print('The best stock in 2001-2018 is:', stock_list[mean_list.index(max_return)], 'It has daily return rate on average:', max_return)
    # return mean_var_list

def infoDemo(stockID):
    print(stockID,'info is shown below.')
    _, _, returnR = data_load(stockID)
    R_np = np.array(returnR)
    sample_var = np.var(R_np)
    plt.hist(returnR, bins = 20) # We expect that "Log return" follows normal.
    plt.xlabel('Monthly return')
    plt.ylabel('Counts (return fall in each bin)')
    plt.title('Visually check the normality assumption')
    plt.show()
    print('Variance of historical return is:', sample_var)
    print('Sample mean of historical return is:', round(np.mean(R_np), 6), '*100% every month [not annulized]')

def BayesPlot(predicted_list, true_list):
    t_list = [i for i in range(len(true_list))]
    plt.plot(t_list, predicted_list, label='Predict')
    plt.plot(t_list, true_list, label = 'Real')
    plt.legend()
    plt.show() 

def main():
    #infoDemo(stockID=stock_list[0])
    stockID = stock_list[1]
    infoDemo(stockID)
    R_df, R_list = BayesPredict(stockID, history=10, acc=6)
    #BayesPlot(R_list, R_df)

time1 = time.time()
main()
time2 = time.time()
print('Running time:', round(time2-time1,4))
