from sympy import *
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
path = './train' # path of data
acc = 10         # the accuracy up to 12 decimal points
def getLogReturn(stockID=0):
    open, close, high, low, vol, adj = read_stock(stockID)
    log_price = [math.log(p) for p in adj]
    log_return = [round(log_price[i+1]-log_price[i],5) for i in range(len(log_price)-1)]
    return log_return

def get_stock_names(path):
    '''input: a path of data; 
    output: a list of data file names'''
    if os.path.isdir(path):
        fileNames = os.listdir(path)
        print('[INFO] found {} stocks'.format(len(fileNames)))
        for i in range(len(fileNames)):
            fileNames[i] = path + '/' + fileNames[i]
    else:
        print('[INFO] Invalid image path')
    return fileNames
    
def read_stock(idx):
    #open, close, high, low = [], [], [], []
    df = pd.read_csv(stock_names[idx])
    open, close, high, low, vol, adj = list(df['Open']), df['Close'], df['High'], df['Low'], df['Volume'], df['Adj Close']
    return open, close, high, low, vol, adj

def Bayes(returnR, sample_var, model='normal', batch_size=10):
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
    x_axis = [0.01*i for i in range(-100,100)]
    last_pdf = [final_belief.subs(miu, i) for i in x_axis]
    prior_pdf = [paraPr.subs(miu,i) for i in x_axis]
    plt.plot(x_axis,last_pdf)
    plt.plot(x_axis,prior_pdf)
    for post in postList[::10]:  # see the evolution of posterior distribution 
        post_pdf = [post.subs(miu,i) for i in x_axis]
        plt.plot(x_axis, post_pdf)
    plt.show()
    if model == 'normal':
        miu_bayes = solve(Eq(log(final_belief).diff(miu), 0), miu)[0] #(lhs: Any, rhs: Any) # print(miu_bayes)# is a list.
    else:
        miu_bayes = solveMax(final_belief, miu, minimumR, maximumR)
    return miu_bayes

def solveMax(func, var, lowerbdd, upperbdd):
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


def demo(stockID=0):
    returnR = getLogReturn(stockID=stockID)
    R_np = np.array(returnR)
    sample_var = np.var(R_np)
    # np.savetxt(str(STOCK_ID)+'.csv', np.array(log_return), delimiter=",")
    plt.hist(returnR, bins=200) # We expect that "Log return" follows normal.
    # plt.plot([i for i in range(len(open))], adj)
    plt.show()
    print('Variance of historical return is:', sample_var)
    bayes_mean = Bayes(returnR, sample_var=sample_var, model='normal', batch_size=100)
    # bayes_mean = Bayes(returnR, sample_var=sample_var, model='t', batch_size=50)
    # bayes_mean = Bayes(returnR, sample_var=sample_var, model='cauchy', batch_size=50)
    print('Sample mean of historical return is:', round(np.mean(R_np), acc), '*100% every day [not annulized]')
    print('Bayesian estimate of mean return is:', round(bayes_mean, acc), '*100% every day [not annulized]')

def getAllmean():
    mean_var_list = []
    mean_list = []
    for i in range(len(stock_names)):
        returnR = getLogReturn(stockID=i)
        R_np = np.array(returnR)
        sample_var = np.var(R_np)
        sample_mean = np.mean(R_np)
        mean_var_list.append([sample_mean, sample_var])
        mean_list.append(sample_mean)
    stock_list = [((stock.split('/'))[-1].split('.'))[0] for stock in stock_names]
    print('All the stocks are listed here:', stock_list)
    df = pd.DataFrame(data=mean_var_list, index=stock_list, columns=['2001-2018 sample mean (RV: daily log-return)', '2001-2018 sample variance (RV: daily log-return)'])
    df.to_csv('Mean-var of daily log-return.csv')
    print('Successfully save in csv.')
    max_return = max(mean_list)
    print('The best stock in 2001-2018 is:', stock_list[mean_list.index(max_return)], 'It has daily return rate on average:', max_return)
    # return mean_var_list

time1 = time.time()
stock_names = get_stock_names(path) # get the names of images # print(stock_names)
demo(0)
time2 = time.time()
getAllmean()
print(time2-time1,'---running time')