# This is the code that execute the data preprocessing and output procedures
# all following packages should have already been installed by PyCharm

import datetime

start_date = datetime.datetime(2018,1,2)

bond_purchase_days = [datetime.datetime(2018,1,2),
             datetime.datetime(2018,7,2),
             datetime.datetime(2019,1,2),
             datetime.datetime(2019,7,1)]

bond_mature_days = [datetime.datetime(2018,6,29),
                    datetime.datetime(2018,12,31),
                    datetime.datetime(2019,6,28),
                    datetime.datetime(2019,12,31)]
#%%
# you can write your code/functions here for carrying out your strategy
def main(current_date, data_stocks, data_test_t, stock_list, in_portfolio, open_price, bond_purchase_days):
    # input: current_date, current date
    #        data_stock, the historical data
    #        data_test_t, the test data up to time current_date
    #        stock_list, the list of stock names
    #        in_portfolio, your portfolio at the end of current_date-1 day
    # you cannot change the above input,
    # you can use the historical data and the test data up to the current date t
    # you can add other input as needed
    # output: buy_action, current_date's buy action
    #         sell_action, current_date's sell action

    # ====================================================================================
    # Your strategy implementation goes here

    # initialize the buy/sell actions
    buy_stock = {}
    sell_stock = {}
    for stock in stock_list:
        buy_stock[stock] = 0.0
        sell_stock[stock] = 0.0

    if not(current_date in bond_purchase_days):
        buy_bond = 0.0
    else:
        # If it is allow to buy bonds on this specific day, enter the amount of money you spend on bonds
        buy_bond = 0.0

    buy_action = {'stock':buy_stock,
                  'bond': buy_bond}
    sell_action = {'stock': sell_stock}

    # ====================================================================================
    # Perform a check on portfolio consistency
    # You should not change this part!!!
    assert sum(open_price[stock] * buy_action['stock'][stock] for stock in stock_list) + buy_action['bond'] <= in_portfolio['cash']
    for stock in stock_list:
        assert in_portfolio['stock'][stock] + buy_action['stock'][stock] >= sell_action['stock'][stock]
    if not(current_date in bond_purchase_days):
        assert buy_action['bond'] == 0.0

    # return the portfolio at the end of the day
    return buy_action, sell_action

# -----------------------------------------------------------------------------------------
# I am providing some templates of trading strategies below

def main_cash(current_date, data_stocks, data_test_t, stock_list, in_portfolio, open_price, bond_purchase_days):
    # input: current_date, current date
    #        data_stock, the historical data
    #        data_test_t, the test data up to time current_date
    #        stock_list, the list of stock names
    #        in_portfolio, your portfolio at the end of current_date-1 day
    # you cannot change the above input,
    # you can use the historical data and the test data up to the current date t
    # you can add other input as needed
    # output: buy_action, current_date's buy action
    #         sell_action, current_date's sell action

    # ====================================================================================
    # Cash strategy: no investment in anything else but holding cash

    # initialize the buy/sell actions
    buy_stock = {}
    sell_stock = {}
    for stock in stock_list:
        buy_stock[stock] = 0.0
        sell_stock[stock] = 0.0
        buy_bond = 0.0

    buy_action = {'stock':buy_stock,
                  'bond': buy_bond}
    sell_action = {'stock': sell_stock}

    # ====================================================================================
    # Perform a check on portfolio consistency
    # You should not change this part!!!
    assert sum(open_price[stock] * buy_action['stock'][stock] for stock in stock_list) + buy_action['bond'] <= in_portfolio['cash']
    for stock in stock_list:
        assert in_portfolio['stock'][stock] + buy_action['stock'][stock] >= sell_action['stock'][stock]
    if not(current_date in bond_purchase_days):
        assert buy_action['bond'] == 0.0

    # return the portfolio at the end of the day
    return buy_action, sell_action

def main_bonds(current_date, data_stocks, data_test_t, stock_list, in_portfolio, open_price, bond_purchase_days):
    # input: current_date, current date
    #        data_stock, the historical data
    #        data_test_t, the test data up to time current_date
    #        stock_list, the list of stock names
    #        in_portfolio, your portfolio at the end of current_date-1 day
    # you cannot change the above input,
    # you can use the historical data and the test data up to the current date t
    # you can add other input as needed
    # output: buy_action, current_date's buy action
    #         sell_action, current_date's sell action

    # ====================================================================================
    # Here I am only buying bonds

    # initialize the buy/sell actions
    buy_stock = {}
    sell_stock = {}
    for stock in stock_list:
        buy_stock[stock] = 0.0
        sell_stock[stock] = 0.0

    if not(current_date in bond_purchase_days):
        buy_bond = 0.0
    else:
        # If it is allow to buy bonds on this specific day, enter the amount of money you spend on bonds
        buy_bond = in_portfolio['cash']

    buy_action = {'stock':buy_stock,
                  'bond': buy_bond}
    sell_action = {'stock': sell_stock}

    # ====================================================================================
    # Perform a check on portfolio consistency
    # You should not change this part!!!
    assert sum(open_price[stock] * buy_action['stock'][stock] for stock in stock_list) + buy_action['bond'] <= in_portfolio['cash']
    for stock in stock_list:
        assert in_portfolio['stock'][stock] + buy_action['stock'][stock] >= sell_action['stock'][stock]
    if not(current_date in bond_purchase_days):
        assert buy_action['bond'] == 0.0

    # return the portfolio at the end of the day
    return buy_action, sell_action

def main_longterm_stock(current_date, data_stocks, data_test_t, stock_list, in_portfolio, open_price, bond_purchase_days):
    # input: current_date, current date
    #        data_stock, the historical data
    #        data_test_t, the test data up to time current_date
    #        stock_list, the list of stock names
    #        in_portfolio, your portfolio at the end of current_date-1 day
    # you cannot change the above input,
    # you can use the historical data and the test data up to the current date t
    # you can add other input as needed
    # output: buy_action, current_date's buy action
    #         sell_action, current_date's sell action

    # ====================================================================================
    # Here I am investing 25000 in DPI, 25000 in BTY, 25000 in GMU, and the rest holding the cash

    # initialize the buy/sell actions
    buy_stock = {}
    sell_stock = {}
    for stock in stock_list:
        buy_stock[stock] = 0.0
        sell_stock[stock] = 0.0
    buy_bond = 0.0

    if current_date == start_date:
        buy_stock['DPI'] = 25000/open_price['DPI']
        buy_stock['BTY'] = 25000/open_price['BTY']
        buy_stock['GMU'] = 25000/open_price['GMU']

    buy_action = {'stock':buy_stock,
                  'bond': buy_bond}
    sell_action = {'stock': sell_stock}

    # ====================================================================================
    # Perform a check on portfolio consistency
    # You should not change this part!!!
    assert sum(open_price[stock] * buy_action['stock'][stock] for stock in stock_list) + buy_action['bond'] <= in_portfolio['cash']
    for stock in stock_list:
        assert in_portfolio['stock'][stock] + buy_action['stock'][stock] >= sell_action['stock'][stock]
    if not(current_date in bond_purchase_days):
        assert buy_action['bond'] == 0.0

    # return the portfolio at the end of the day
    return buy_action, sell_action