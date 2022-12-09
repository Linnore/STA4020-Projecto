# -----------------------------------------------------------------------------------------
# Do not change anything below!!!!!

import csv, os, datetime

start_date = datetime.datetime(2018,1,2)

bond_purchase_days = [datetime.datetime(2018,1,2),
             datetime.datetime(2018,7,2),
             datetime.datetime(2019,1,2),
             datetime.datetime(2019,7,1)]

bond_mature_days = [datetime.datetime(2018,6,29),
                    datetime.datetime(2018,12,31),
                    datetime.datetime(2019,6,28),
                    datetime.datetime(2019,12,31)]

from main import *

def input_csv(file_path):
    # read the .csv file containing one stock and output the dictionary
    # input: the file path pointing to the CSV file
    # output: the dictionary that contains the data for the stock specified by .csv file
    data_stock = {'date': [],
                  'open': [],
                  'high': [],
                  'low': [],
                  'close': [],
                  'adjusted_close': [],
                  'volume': []
                  }
    with open(file_path, 'r') as stock_file:
        csvReader = csv.reader(stock_file)
        iRow = 0
        for item in csvReader:
            if iRow == 0:
                title = item
                iRow += 1
            else:
                try:
                    data_stock['date'].append(datetime.datetime.strptime(item[title.index('Date')], '%Y-%m-%d'))
                    data_stock['open'].append(float(item[title.index('Open')]))
                    data_stock['high'].append(float(item[title.index('High')]))
                    data_stock['low'].append(float(item[title.index('Low')]))
                    data_stock['close'].append(float(item[title.index('Close')]))
                    data_stock['adjusted_close'].append(float(item[title.index('Adj Close')]))
                    data_stock['volume'].append(float(item[title.index('Volume')]))
                except:
                    print(file_path, item)
    return data_stock

def input_files(folder_path):
    # examine the folder, obtain the list of csv files and the input the data
    # input: folder_path, e.g. '/Users/MFE5100/Project/' or 'C:\\MFE5100\\Project'
    # output: the dictionary contains all stock data, each key represents a stock
    filesList = os.listdir(folder_path)
    data_stocks = {}
    stock_list = []
    # read all .csv in the path 'folder_path'
    for item in filesList:
        if '.csv' in item:
            file_path = os.path.join(folder_path, item)
            stock_name = item[:-4]
            stock_list.append(stock_name)
            data_stocks[stock_name] = input_csv(file_path)
    return data_stocks, stock_list

def excerpt_t(data_test, t):
    # obtain the data for test
    # input: data_test, the input data for test, which has the same format as data_stock
    #        t, the date the data is up to, in datetime.date format
    # output: the test data up to time t
    data_test_t = {}
    for stock in data_test.keys():
        time_index = data_test[stock]['date'].index(t)
        data_test_t[stock] = {
            'date': data_test[stock]['date'][:time_index],
            'open': data_test[stock]['open'][:time_index],
            'high': data_test[stock]['high'][:time_index],
            'low': data_test[stock]['low'][:time_index],
            'close': data_test[stock]['close'][:time_index],
            'adjusted_close': data_test[stock]['adjusted_close'][:time_index],
            'volume': data_test[stock]['volume'][:time_index]
        }
    return data_test_t

def testing(data_stocks, data_test, stock_list, dateList, bond_purchase_days):
    # the main strategy to be implemented
    # input: data_stocks, the data read from the .csv file
    #        data_test, the data read from the testing .csv file
    # output: the final value of the portfolio
    initial_stock = {}
    for stock in stock_list:
        initial_stock[stock] = 0.0
    initial_bond = 0.0
    initial_cash = 100000.0
    portfolio = [{
        'stock': initial_stock,
        'bond': initial_bond,
        'cash': initial_cash
    }]
    for t in range(len(dateList)):
        # here I will call your function "main" to get the your output portfolio
        current_date = dateList[t]
        current_portfolio = portfolio[t].copy()
        data_test_t = excerpt_t(data_test, current_date)
        open_price = {}
        close_price = {}
        for stock in stock_list:
            open_price[stock] = data_test[stock]['open'][t]
            close_price[stock] = data_test[stock]['close'][t]
        buy_action, sell_action = main(current_date, data_stocks, data_test_t, stock_list, current_portfolio, open_price, bond_purchase_days)
        # buy_action, sell_action = main_cash(current_date, data_stocks, data_test_t, stock_list, current_portfolio, open_price, bond_purchase_days)
        # buy_action, sell_action = main_bonds(current_date, data_stocks, data_test_t, stock_list, current_portfolio, open_price, bond_purchase_days)
        # buy_action, sell_action = main_longterm_stock(current_date, data_stocks, data_test_t, stock_list, current_portfolio, open_price, bond_purchase_days)

        # calculating the portfolio at the beginning of the next day
        next_portfolio = current_portfolio.copy()
        # calculate the stock position
        for stock in stock_list:
            next_portfolio['stock'][stock] = current_portfolio['stock'][stock] + buy_action['stock'][stock] - sell_action['stock'][stock]
        # calculate the cash position
        after_buy_cash = current_portfolio['cash'] - buy_action['bond'] - sum(buy_action['stock'][stock] * open_price[stock] for stock in stock_list)
        next_portfolio['cash'] = after_buy_cash * 1.00005 + sum(sell_action['stock'][stock]*close_price[stock] for stock in stock_list)
        # calculate the bond position
        if current_date in bond_mature_days:
            next_portfolio['bond'] = 0.0
            next_portfolio['cash'] += current_portfolio['bond'] * 1.014889156509222
        elif current_date in bond_purchase_days:
            next_portfolio['bond'] += buy_action['bond']
        portfolio.append(next_portfolio)

    # calculate the last day's portfolio value
    close_price_final = {}
    t_final = len(dateList) - 1
    for stock in stock_list:
        close_price_final[stock] = data_test[stock]['close'][t_final]
    portfolio_value = portfolio[-1]['bond'] + portfolio[-1]['cash'] + \
                      sum(portfolio[-1]['stock'][stock] * close_price_final[stock] for stock in stock_list)
    return portfolio_value

data_stocks, stock_list_train = input_files('./train/')
data_test, stock_list_test = input_files('./sample_test/')
assert stock_list_test == stock_list_train
stock_list = stock_list_test
dateList = data_test['SNE']['date']