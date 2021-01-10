# -*- coding: utf-8 -*
#!/usr/bin/env python3

import socket
from socket import AF_INET, SOCK_STREAM
import threading 
import queue

import json
import sys
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import talib
import numpy as np
import time

from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

from sqlalchemy import Column, ForeignKey, Integer, Float, String
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import inspect
from sqlalchemy import and_


from flask import Flask, render_template
app = Flask(__name__, template_folder='templates')



clientID = "yicheng"


' download data '
data_start_date = dt.datetime(2014,1,1)  # hours:minute:seconds
data_end_date = dt.date.today()  # only dates
requestURL = "https://eodhistoricaldata.com/api/eod/"
myEodKey = "5ba84ea974ab42.45160048"
requestSP500 = "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_json/data/64dd3e9582b936b0352fdd826ecd3c95/constituents_json.json"

' trading '
engine = create_engine('sqlite:///pairs_trading.db')
engine.execute("PRAGMA foreign_keys = ON")
metadata = MetaData()
metadata.reflect(bind=engine)  # bind to Engine, load all tables

' Parameters '
training_start_date = dt.datetime(2014,1,1)
training_end_date = dt.datetime(2018,1,1)
backtesting_start_date = dt.datetime(2018,1,1)
backtesting_end_date = dt.datetime(2019,1,1)
capital = 1000000.
significance = 0.05
k = 2
mvt = 10
# PCA
N_PRIN_COMPONENTS = 50
epsilon = 1.8



def get_daily_data(symbol='', start=data_start_date, end=data_end_date, requestType=requestURL, 
                   apiKey=myEodKey, completeURL=None):
    if not completeURL:
        symbolURL = str(symbol) + '?'
        startURL = "from=" + str(start)
        endURL = "to=" + str(end)
        apiKeyURL = "api_token=" + myEodKey
        completeURL = requestURL + symbolURL + startURL + '&' + endURL + '&' + apiKeyURL + '&period=d&fmt=json'
    
    # if cannot open url
    try:
        with urllib.request.urlopen(completeURL) as req:
            data = json.load(req)
            return data
    except:
        pass
    

' populate stock data for each stock '
def download_stock_data(ticker, metadata, engine, table_name):
    column_names = ['symbol','date','open','high','low','close','adjusted_close','volume']
    price_list = []
    clear_a_table(table_name, metadata, engine)
    
    if 'GSPC' not in ticker:
        symbol_full = str(ticker) + ".US"
        stock = get_daily_data(symbol=symbol_full)
    else:
        stock = get_daily_data(symbol=ticker)

    if stock:
        for stock_data in stock:
            price_list.append([str(ticker), stock_data['date'], stock_data['open'], stock_data['high'],
                           stock_data['low'], stock_data['close'], stock_data['adjusted_close'],
                           stock_data['volume']])
    
    stocks = pd.DataFrame(price_list, columns=column_names)
    stocks.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=5)


def execute_sql_statement(sql_st, engine):
    result = engine.execute(sql_st)
    result_df = pd.DataFrame(result.fetchall())
    result_df.columns = result.keys()
    return result_df  


''' create table '''
def create_sp500_info_table(name, metadata, engine, null=False):
    table = Table(name, metadata, 
                  Column('name', String(50), nullable=null),
                  Column('sector', String(50), nullable=null),
                  Column('symbol', String(50), primary_key=True, nullable=null),
                  extend_existing = True)  # constructor
    table.create(engine, checkfirst=True)
    
def create_price_table(name, metadata, engine, null=True):
    if name != 'GSPC.INDX':
        foreign_key = 'sp500.symbol'
        table = Table(name, metadata, 
                    Column('symbol', String(50), ForeignKey(foreign_key), 
                           primary_key=True, nullable=null),
                    Column('date', String(50), primary_key=True, nullable=null),
                    Column('open', Float, nullable=null),
                    Column('high', Float, nullable=null),
                    Column('low', Float, nullable=null),
                    Column('close', Float, nullable=null),
                    Column('adjusted_close', Float, nullable=null),
                    Column('volume', Integer, nullable=null),
                    extend_existing = True)
    else:
        table = Table(name, metadata, 
            Column('symbol', String(50), primary_key=True, nullable=null),
            Column('date', String(50), primary_key=True, nullable=null),
            Column('open', Float, nullable=null),
            Column('high', Float, nullable=null),
            Column('low', Float, nullable=null),
            Column('close', Float, nullable=null),
            Column('adjusted_close', Float, nullable=null),
            Column('volume', Integer, nullable=null),
            extend_existing = True)
    table.create(engine, checkfirst=True)

def create_stockpairs_table(table_name, metadata, engine):
    table = Table(table_name, metadata,
                  Column('Ticker1', String(50), primary_key=True, nullable=False),
                  Column('Ticker2', String(50), primary_key=True, nullable=False),
                  Column('Score', Float, nullable=False),
                  Column('Profit_Loss', Float, nullable=False),
                  extend_existing=True)
    table.create(engine, checkfirst=True)

def create_pairprices_table(table_name, metadata, engine, null=True):
    table = Table(table_name, metadata,
                  Column('Symbol1', String(50), ForeignKey('stockpairs.Ticker1'), primary_key=True, nullable=null),
                  Column('Symbol2', String(50), ForeignKey('stockpairs.Ticker2'), primary_key=True, nullable=null),
                  Column('Date', String(50), primary_key=True, nullable=null),
                  Column('Close1', Float, nullable=null),
                  Column('Close2', Float, nullable=null),
                  Column('Residual', Float, nullable=null),
                  Column('Lower', Float, nullable=null),
                  Column('MA', Float, nullable=null),
                  Column('Upper', Float, nullable=null),
                  extend_existing=True)
    table.create(engine, checkfirst=True)
    
def create_trades_table(table_name, metadata, engine, null=False):
    table = Table(table_name, metadata,
                  Column('Symbol1', String(50), ForeignKey('stockpairs.Ticker1'), primary_key=True, nullable=null),
                  Column('Symbol2', String(50), ForeignKey('stockpairs.Ticker2'), primary_key=True, nullable=null),
                  Column('Date', String(50), primary_key=True, nullable=null),
                  Column('Close1', Float, nullable=null),
                  Column('Close2', Float, nullable=null),
                  Column('Qty1', Float, nullable=null),
                  Column('Qty2', Float, nullable=null),
                  Column('P/L', Float, nullable=null),
                  extend_existing=True)
    table.create(engine, checkfirst=True)

def clear_a_table(table_name, metadata, engine):
    conn = engine.connect()
    table = metadata.tables[table_name]
    delete_st = table.delete()
    conn.execute(delete_st)


def download_market_data(metadata, engine, sp500_info_df):
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Downloading data ...")
    
    ' put sp500 constituent data into databases '
    create_sp500_info_table('sp500', metadata, engine)
    clear_a_table('sp500', metadata, engine)  # clear table before insert
    sp500_info_df.to_sql('sp500', con=engine, if_exists='append', index=False,
                         chunksize=5)

    ' get data for each ticker from sp500 '
    for symbol in sp500_info_df.Symbol:
        create_price_table(symbol, metadata, engine)
        download_stock_data(symbol, metadata, engine, symbol)
    
    ' SP500 index price '
    create_price_table('GSPC.INDX', metadata, engine)
    download_stock_data('GSPC.INDX', metadata, engine, 'GSPC.INDX')
    
    print("Finished downloading.")


def training_data(metadata, engine, significance, sp500_info_df,
                  training_start_date, training_end_date): 
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Training data ...")
    print("Start date:", training_start_date, ", End date:", training_end_date)
    
    ' get training set '
    Price = pd.DataFrame()
    
    for symbol in sp500_info_df.Symbol:
        select_st = "SELECT date, adjusted_close From " + "\"" + symbol + "\"" +  \
        " WHERE date >= " + "\"" + str(training_start_date) + "\"" +     \
        " AND date <= " + "\"" + str(training_end_date) + "\"" + ";"
        try:
            result_df = execute_sql_statement(select_st, engine)
            result_df.set_index('date', inplace=True)  # date as index
            result_df.columns = [symbol]  # name is column
            Price = pd.concat([Price, result_df], axis=1, sort=True)
        except:
            pass
            
    ' PCA: reduce dimension '
    Price.sort_index(inplace=True)
    Price.fillna(method='ffill', inplace=True)
    Price = Price.loc[:,(Price>0).all(0)]  # every price > 0
    
    Price_ret = Price.pct_change()
    Price_ret = Price_ret.replace([np.inf, -np.inf], np.nan)
    Price_ret.dropna(axis=0, how='all', inplace=True) # drop first row (NA)
    Price_ret.dropna(axis=1, how='any', inplace=True)
    
    pca = PCA(n_components=N_PRIN_COMPONENTS)
    pca.fit(Price_ret)
    X = pd.DataFrame(pca.components_.T, index=Price_ret.columns)
    sp500_info_df.set_index('Symbol', inplace=True)
    X = pd.concat([X, sp500_info_df.Sector.T], axis=1, sort=True)
    X = pd.get_dummies(X)
    
    ' DBSCAN: identify clusters from stocks that are closest '
    X.dropna(axis=0, how='any', inplace=True)
    X_arr = preprocessing.StandardScaler().fit_transform(X)
    clf = DBSCAN(eps=epsilon, min_samples=3)
    
    # labels is label values from -1 to x
    # -1 represents noisy samples that are not in clusters
    clf.fit(X_arr)
    clustered = clf.labels_
    # all stock with its cluster label (including -1)
    clustered_series = pd.Series(index=X.index, data=clustered.flatten())
    # clustered stock with its cluster label
    clustered_series = clustered_series[clustered_series != -1]
    
    poss_cluster = clustered_series.value_counts().sort_index()
    print(poss_cluster)
    
    'identify cointegrated pairs from clusters'
    def Cointegration(cluster, significance, start_day, end_day):
        pair_coin = []
        p_value = []
        adf = []
        n = cluster.shape[0]
        keys = cluster.keys()
        for i in range(n):
            for j in range(i+1,n):
                asset_1 = Price.loc[start_day:end_day, keys[i]]
                asset_2 = Price.loc[start_day:end_day, keys[j]]
                results = sm.OLS(asset_1, asset_2)
                results = results.fit()
                predict = results.predict(asset_2)
                error = asset_1 - predict
                ADFtest = ts.adfuller(error)
                if ADFtest[1] < significance:
                    pair_coin.append([keys[i], keys[j]])  # pair names
                    p_value.append(ADFtest[1])  # p value, smaller the better
                    adf.append(ADFtest[0])  # adf test stats, larger the better
        return p_value, pair_coin, adf
    
    "Pair selection method"
    "select a pair with lowest p-value from each cluster"
    def PairSelection(clustered_series, significance, 
                      start_day=str(training_start_date), end_day=str(training_end_date)):
        Opt_pairs = []   # to get best pair in cluster i
        tstats = []

        for i in range(len(poss_cluster)):
            cluster = clustered_series[clustered_series == i]
            result = Cointegration(cluster, significance, start_day, end_day)
            if len(result[0]) > 0:
                if np.min(result[0]) < significance:
                    index = np.where(result[0] == np.min(result[0]))[0][0]
                    Opt_pairs.append([result[1][index][0], result[1][index][1]])
                    tstats.append(round(result[2][index], 4))
        
        return Opt_pairs, tstats
    
    stock_pairs, tstats = PairSelection(clustered_series, significance)
    # put into sql table
    create_stockpairs_table('stockpairs', metadata, engine)
    clear_a_table('stockpairs', metadata, engine)
    stock_pairs = pd.DataFrame(stock_pairs, columns=['Ticker1', 'Ticker2'])
    stock_pairs["Score"] = -1 * np.array(tstats)
    stock_pairs["Profit_Loss"] = 0.0
    stock_pairs.to_sql('stockpairs', con=engine, if_exists='append', index=False, chunksize=5)
    
    print(stock_pairs[["Ticker1", "Ticker2"]])
    print("Finished training.")
    return stock_pairs


def building_model(metadata, engine, k, mvt,
                   backtesting_start_date, backtesting_end_date):
    global ols_results
    '''
    get pair prices, moving averages, bollinger bands
    k: number of std
    mvt: moving average period
    '''
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Building Model ...")
    print("Parameters: k =", k, ", moving average =", mvt)
    
    select_st = "SELECT Ticker1, Ticker2 from stockpairs;"
    stock_pairs = execute_sql_statement(select_st, engine)
    
    create_pairprices_table('pairprices', metadata, engine, mvt)
    clear_a_table('pairprices', metadata, engine)
    
    for pair in stock_pairs.values:
        select_st = "SELECT stockpairs.Ticker1 as Symbol1, stockpairs.Ticker2 as Symbol2, \
                     " + pair[0] + ".date as Date, " + pair[0] + ".Adjusted_close as Close1, \
                     " + pair[1] + ".Adjusted_close as Close2 \
                     From " + pair[0] + ", " + pair[1] + ", stockpairs \
                     Where (((stockpairs.Ticker1 = " + pair[0] + ".symbol) and \
                     (stockpairs.Ticker2 = " + pair[1] + ".symbol)) and \
                    (" + pair[0] + ".date = " + pair[1] + ".date)) \
                    and " + pair[0] + ".date >= " + "\"" + str(training_start_date) + "\"" +   \
                    " AND " + pair[0] + ".date <= " + "\"" + str(training_end_date) + "\"  \
                    ORDER BY Symbol1, Symbol2;"
                     
        result_df = execute_sql_statement(select_st, engine)
        
        select_st = "SELECT stockpairs.Ticker1 as Symbol1, stockpairs.Ticker2 as Symbol2, \
                     " + pair[0] + ".date as Date, " + pair[0] + ".Adjusted_close as Close1, \
                     " + pair[1] + ".Adjusted_close as Close2 \
                     FROM " + pair[0] + ", " + pair[1] + ", stockpairs \
                     WHERE (((stockpairs.Ticker1 = " + pair[0] + ".symbol) and \
                     (stockpairs.Ticker2 = " + pair[1] + ".symbol)) and \
                     (" + pair[0] + ".date = " + pair[1] + ".date)) \
                     and " + pair[0] + ".date >= " + "\"" + str(backtesting_start_date) + "\"" +   \
                     " AND " + pair[0] + ".date <= " + "\"" + str(backtesting_end_date) + "\"  \
                     ORDER BY Symbol1, Symbol2;"          
        result_df2 = execute_sql_statement(select_st, engine)
        
        # get bollinger band
        results = sm.OLS(result_df.Close1, sm.add_constant(result_df.Close2)).fit()
        predict = results.params[0] + results.params[1] * result_df2.Close2
        ols_results[pair[0]] = results
        error = np.subtract(result_df2.Close1, predict)
        upperband, middleband, lowerband = talib.BBANDS(error, timeperiod=mvt, 
                                                  nbdevup=k, nbdevdn=k, matype=0)
        result_df2[['Residual', 'Lower', 'MA', 'Upper']] = pd.DataFrame([error, lowerband, middleband, upperband]).T.round(4)
        result_df2.to_sql('pairprices', con=engine, if_exists='append', index=False, chunksize=5)
    
    print("Finished building model.")
    

class StockPair:

    def __init__(self, symbol1, symbol2, start_date, end_date):
        self.ticker1 = symbol1
        self.ticker2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.trades = {}
        self.total_profit_loss = 0.0
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__) + "\n"
    
    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__) + "\n"
    
    def createTrade(self, date, close1, close2, res, lower, upper, qty1 = 0, qty2 = 0, profit_loss = 0.0):
        self.trades[date] = np.array([close1, close2, res, lower, upper, qty1, qty2, profit_loss])
        
    def updateTrades(self):  # dollar neutral, available dollar for buy/sell for each pair
        trades_matrix = np.array(list(self.trades.values()))
        
        for index in range(1, trades_matrix.shape[0]):
            # RES SELL SIGNAL: buy asset 1, sell asset 2
            if (trades_matrix[index-1, 2] < trades_matrix[index-1, 4] and 
                trades_matrix[index, 2] > trades_matrix[index, 4]):
                trades_matrix[index, 5] = int(capital / trades_matrix[index, 0])
                trades_matrix[index, 6] = int(-capital / trades_matrix[index, 1])
            # RES BUY SIGNAL: sell asset 1, buy asset 2
            elif (trades_matrix[index-1, 2] > trades_matrix[index-1, 3] and 
                trades_matrix[index, 2] < trades_matrix[index, 3]):
                trades_matrix[index, 5] = int(-capital / trades_matrix[index, 0])
                trades_matrix[index, 6] = int(capital / trades_matrix[index, 1])
            # no act
            else:
                trades_matrix[index, 5] = trades_matrix[index-1, 5]
                trades_matrix[index, 6] = trades_matrix[index-1, 6]
                
            'update profit and loss'
            trades_matrix[index, 7] = trades_matrix[index, 5] * (trades_matrix[index, 0] - trades_matrix[index-1, 0])     \
                                    + trades_matrix[index, 6] * (trades_matrix[index, 1] - trades_matrix[index-1, 1])
            trades_matrix[index, 7] = round(trades_matrix[index, 7], 2)
            self.total_profit_loss += trades_matrix[index, 7]
            
        for key, index in zip(self.trades.keys(), range(0, trades_matrix.shape[0])):
            self.trades[key] = trades_matrix[index]
            
        return pd.DataFrame(trades_matrix[:, range(5, trades_matrix.shape[1])], columns=['Qty1', 'Qty2', 'P/L'])    
 

def back_testing(metadata, engine, backtesting_start_date, backtesting_end_date):
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Backtesting ...")
    print("Start date:", backtesting_start_date, ", End date:", backtesting_end_date)
    
    print('create StockPair')
    stock_pair_map = dict()
    
    select_st = 'SELECT Ticker1, Ticker2 FROM stockpairs;'
    stock_pairs = execute_sql_statement(select_st, engine)
    
    for index, row in stock_pairs.iterrows():
        aKey = (row['Ticker1'], row['Ticker2'])
        stock_pair_map[aKey] = StockPair(row['Ticker1'], row['Ticker2'], 
                                  backtesting_start_date, backtesting_end_date)

    print('create Trades')
    select_st = 'SELECT * FROM pairprices;'
    result_df = execute_sql_statement(select_st, engine)
        
    for index in range(result_df.shape[0]):
        aKey = (result_df.at[index, 'Symbol1'], result_df.at[index, 'Symbol2'])
        stock_pair_map[aKey].createTrade(result_df.at[index, 'Date'], 
                      result_df.at[index, 'Close1'], result_df.at[index, 'Close2'],
                      result_df.at[index, 'Residual'], result_df.at[index, 'Lower'],
                      result_df.at[index, 'Upper'])
    
    print('update Trades')
    trades_df = pd.DataFrame(columns=['Qty1', 'Qty2', 'P/L'])
    for key, value in stock_pair_map.items():
        trades_df = trades_df.append(value.updateTrades(), ignore_index=True)
    
        table = metadata.tables['stockpairs']
        update_st = table.update().values(Profit_Loss=value.total_profit_loss).where( \
                and_(table.c.Ticker1==value.ticker1, table.c.Ticker2==value.ticker2))
        engine.execute(update_st)
        
    result_df = result_df[['Symbol1', 'Symbol2', 'Date', 'Close1', 'Close2']].join(trades_df)
    
    create_trades_table('trades', metadata, engine)
    clear_a_table('trades', metadata, engine)
    result_df.to_sql('trades', con=engine, if_exists='append', index=False, chunksize=5)
    
    print("Finished backtesting.")


'real time data according to market date'
def feed_realtime_data(ticker, start, end):
    global price_data
    column_names = ['symbol','date','adjusted_close']
    
    stock = get_daily_data(symbol=ticker, start=start, end=end)
    if stock:
        for stock_data in stock:
            price_data.append([str(ticker), stock_data['date'], 
                               stock_data['adjusted_close']])
    stocks = pd.DataFrame(price_data, columns=column_names)
    stocks.adjusted_close = stocks.adjusted_close.astype(float)
    return stocks


def get_orders(market_date=None):
    orders_list = []
    
    select_st = 'SELECT Ticker1, Ticker2 FROM stockpairs;'
    pairs = execute_sql_statement(select_st, engine)

    for index, row in pairs.iterrows():
        # previous data for ols fit
        select_st = "SELECT symbol, date, adjusted_close FROM "+str(row[0])+    \
                    " WHERE date >= " + "\"" + str(backtesting_start_date) + "\"" +     \
                    " AND date <= " + "\"" + str(backtesting_end_date) + "\"" + ";"
        result1 = execute_sql_statement(select_st, engine)
        select_st = "SELECT symbol, date, adjusted_close FROM "+str(row[1])+    \
                    " WHERE date >= " + "\"" + str(backtesting_start_date) + "\"" +     \
                    " AND date <= " + "\"" + str(backtesting_end_date) + "\"" + ";"
        result2 = execute_sql_statement(select_st, engine)

        if market_date:
            # append latest real data to previous data
            stock1 = feed_realtime_data(row[0], market_date, market_date)
            stock1 = stock1[stock1.symbol == row[0]]
            result1 = pd.concat([result1, stock1], ignore_index=True)
            stock2 = feed_realtime_data(row[1], market_date, market_date)
            stock2 = stock2[stock2.symbol == row[1]]
            result2 = pd.concat([result2, stock2], ignore_index=True)
        
        try:            
            results = ols_results[row[0]]
            predict = results.params[0] + results.params[1] * result2.adjusted_close
            error = np.subtract(result1.adjusted_close, predict)
            upperband, middleband, lowerband = talib.BBANDS(error, timeperiod=mvt, 
                                                      nbdevup=k, nbdevdn=k, matype=0)
            price1 = round(result1.adjusted_close.values[-1], 2)
            price2 = round(result2.adjusted_close.values[-1], 2)
            
            if (error.values[-2] < upperband.values[-2] and error.values[-1] > upperband.values[-1]):
                amt1 = int(capital / price1)
                amt2 = int(capital / price2)
                order1 = 'Order New '+row[0]+' Buy '+str(price1)+' '+str(amt1)
                order2 = 'Order New '+row[1]+' Sell '+str(price2)+' '+str(amt2)
                orders_list.append(order1)
                orders_list.append(order2)
                print(order1, ',', order2)
    
            elif error.values[-2] > lowerband.values[-2] and error.values[-2] < lowerband.values[-1]:
                amt1 = int(capital / price1)
                amt2 = int(capital / price2)
                order1 = 'Order New '+row[0]+' Sell '+str(price1)+' '+str(amt1)
                order2 = 'Order New '+row[1]+' Buy '+str(price2)+' '+str(amt2)
                orders_list.append(order1)
                orders_list.append(order2)
                print(order1, ',', order2)
                
            else:
                print(row[0], row[1], 'No order signal.')
        
        except:
            print('No order signal.')
            
    return orders_list

    
def receive(e, q):
    """Handles receiving of messages."""
    total_server_response = []
    msg_end_tag = ".$$$$"
    
    while True:
        try:
            recv_end = False
            # everytime only load certain size
            server_response = client_socket.recv(BUFSIZ).decode("utf8")
            
            if server_response:
                if msg_end_tag in server_response:  # if reaching end of message
                    server_response = server_response.replace(msg_end_tag, '')
                    recv_end = True
                
                # append every response
                total_server_response.append(server_response)
                
                # if reaching the end, put it into queue
                if recv_end == True:
                    server_response_message = ''.join(total_server_response)
                    data = json.loads(server_response_message)
                    #print(data)
                    q.put(data)
                    total_server_response = []
                    
                    if e.isSet():
                        e.clear()
                        
        except OSError:  # Possibly client has left the chat.
            break
        

' The logon message includes the list of stocks from client '
def get_stock_list_from_database():
    select_st = 'SELECT Ticker1, Ticker2 FROM stockpairs;'
    pairs = execute_sql_statement(select_st, engine)
    tickers = pd.concat([pairs["Ticker1"], pairs["Ticker2"]], ignore_index=True)
    tickers.drop_duplicates(keep='first', inplace=True)
    tickers.sort_values(axis=0, ascending=True, inplace=True, kind='quicksort')
    print(tickers)
    return tickers

def logon():
    tickers = get_stock_list_from_database();
    client_msg = json.dumps({'Client':clientID, 'Status':'Logon', 'Stocks':tickers.str.cat(sep=',')})
    return client_msg

def get_user_list():
    client_msg = "{\"Client\":\"" + clientID + "\", \"Status\":\"User List\"}"
    return client_msg
    
def get_stock_list():
    client_msg = "{\"Client\":\"" + clientID + "\", \"Status\":\"Stock List\"}"
    return client_msg

def get_market_status():
    client_msg = json.dumps({'Client':clientID, 'Status':'Market Status'})
    return client_msg

def get_order_table(stock_list):
    client_msg = json.dumps({'Client':clientID, 'Status':'Order Inquiry', 'Symbol':stock_list})
    return client_msg
    
def enter_a_new_order(symbol, side, price, qty):
    client_msg = json.dumps({'Client':clientID, 'Status':'New Order', 'Symbol':symbol, 'Side':side, 'Price':price, 'Qty':qty})
    return client_msg

def quit_connection():
    client_msg = "{\"Client\":\"" + clientID + "\", \"Status\":\"Quit\"}"
    return client_msg

def send_msg(client_msg):
    client_socket.send(bytes(client_msg, "utf8"))
    data = json.loads(client_msg)
    return data

def set_event(e):
    e.set();
    
def wait_for_an_event(e):
    while e.isSet():
        continue

def get_data(q):
    data = q.get()
    q.task_done()
#    print(dt.datetime.now(), data)
    return data


# command in queue
def join_trading_network(e, q):
    global market_period_list, record_order_df
    last_close_time = time.time()
    
    threading.Thread(target=receive, args=(e,q)).start()
    
    set_event(e)
    send_msg(logon())  # automatic logon
    wait_for_an_event(e)
    get_data(q)
    
    set_event(e)
    send_msg(get_user_list())  # automatic print out user list
    wait_for_an_event(e)
    get_data(q)
    
    set_event(e)
    send_msg(get_stock_list())  # automatically print out stock list
    wait_for_an_event(e)
    get_data(q)
    
    while True:
        set_event(e)
        client_msg = get_market_status()  # automatically print market status
        send_msg(client_msg)
        wait_for_an_event(e)
        data = get_data(q)
        market_status = data["Market Status"]
         
        'The client will loop until market open'
        if (market_status == "Market Closed" or
            market_status == "Pending Open" or
            market_status == "Not Open"):
            # if market closed too long, stop trading
            if time.time() - last_close_time > 150:
                print('>>>> Stop trading after ', time.time() - last_close_time, 'seconds') 
                break;
            time.sleep(1)
            continue
        
        last_close_time = time.time()
        
        ' place order every 40s (1day) '
        print('======================================================')        
        market_period = data["Market Period"]
        market_period_list.append(market_period)  # store past dates
        print("Current market status is:", market_status)
        print("Market period is:", market_period_list)
        
        ' pLace order according to strategy using previous close price'
        if len(market_period_list) > 1:
            prev_date = market_period_list[-2]
            orders_list = get_orders(prev_date)  # up to previous day close price
        else:
            orders_list = get_orders()
        
        'The client will send orders to server only during market open and pending closing'
        if orders_list:
            
            for order in orders_list:
                order_list = order.split(" ")
                mySymbol = str(order_list[2])
                mySide = str(order_list[3])
                myPrice = float(order_list[4])
                myQuantity = int(order_list[5])

                set_event(e)
                send_msg(get_order_table([mySymbol]))  # pass in list
                wait_for_an_event(e)
                data = get_data(q)
                order_data = json.loads(data) 
                order_table = pd.DataFrame(order_data["data"])
                if order_table.empty:
                    print('Empty table')
                    continue
                
                if mySide == 'Buy':
                    order_table = order_table[order_table["Side"] == 'Sell']
                    order_table.sort_values('Price', ascending=True, inplace=True)
                    order_table.reset_index(drop=True, inplace=True)
                    best_price = order_table.loc[0, 'Price']
                    order_index = order_table.loc[0, 'OrderIndex']
                else:
                    order_table = order_table[order_table["Side"] == 'Buy']
                    order_table.sort_values('Price', ascending=False, inplace=True)
                    order_table.reset_index(drop=True, inplace=True)
                    best_price = order_table.loc[0, 'Price']
                    order_index = order_table.loc[0, 'OrderIndex']
                print(order_table.iloc[0, :])
                print('today best price', best_price, ', previous day close price', myPrice, ', order index', order_index)
                
                set_event(e)
                client_msg = enter_a_new_order(symbol=mySymbol, side=mySide, price=float(best_price), qty=myQuantity)
                send_msg(client_msg)
                wait_for_an_event(e)
                data = get_data(q)
                
                'record orders'
                record_order = pd.Series([market_period, mySymbol, mySide, best_price, myQuantity])
                record_order_df = pd.concat([record_order_df, record_order], axis=1)
        
        time.sleep(30)  # skip to next day
    
    record_order_df = record_order_df.T
    try:
        record_order_df.columns = ['Date', 'Symbol', 'Side', 'Price', 'Quantity']
        record_order_df.loc[record_order_df['Side']=='Sell', 'Quantity'] = -1.*record_order_df.loc[record_order_df['Side']=='Sell', 'Quantity']
        record_order_df.set_index(['Symbol', 'Date'], inplace=True)
        print(record_order_df)
    except:
        print('No Orders!!!!')
    
    
    set_event(e)
    send_msg(quit_connection())  # automatically quit
    wait_for_an_event(e)


    
'define function to calculate maximum drawdown'
def MaxDrawdown(Ret_Cum):
    # ret_cum also can be portfolio position series
    ContVal = np.zeros(np.size(Ret_Cum))
    MaxDD = np.zeros(np.size(Ret_Cum))
    for i in range(np.size(Ret_Cum)):
        if i == 0:
            if Ret_Cum[i] < 0:
                ContVal[i] = Ret_Cum[i]
            else:
                ContVal[i] = 0
        else:
            ContVal[i] = Ret_Cum[i] - np.nanmax(Ret_Cum[0:(i+1)])
        MaxDD[i] = np.nanmin(ContVal[0:(i+1)])
    return MaxDD
    

@app.route('/')
def index():
    return render_template("index.html")

  
@app.route('/data_prep')
def data_prep():
    inspector = inspect(engine)
    
    sp500_info = get_daily_data(completeURL=requestSP500)
    sp500_info_df = pd.DataFrame(sp500_info)
    if len(inspector.get_table_names()) == 0:  # if no market data, download market data
        download_market_data(metadata, engine, sp500_info_df)
    else:
        print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Data already downloaded ...")
        
    stock_pairs = training_data(metadata, engine, significance, sp500_info_df,
                                training_start_date, training_end_date)
    pairs = stock_pairs.transpose()
    list_of_pairs = [pairs[i] for i in pairs]
    return render_template("data_prep.html", pair_list=list_of_pairs)


@app.route('/build_model')
def build_model():
    building_model(metadata, engine, k, mvt, 
                   backtesting_start_date, backtesting_end_date)
    
    select_st = "SELECT * from pairprices;"
    result_df = execute_sql_statement(select_st, engine)
    result_df = result_df.transpose()
    list_of_pairs = [result_df[i] for i in result_df]
    return render_template("build_model.html", pair_list=list_of_pairs)


@app.route('/back_test')
def model_back_testing():
    back_testing(metadata, engine, backtesting_start_date, backtesting_end_date)
    
    select_st = "SELECT * from stockpairs;"
    result_df = execute_sql_statement(select_st, engine)
    result_df['Score'] = result_df['Score'].map('{:.4f}'.format)
    result_df['Profit_Loss'] = result_df['Profit_Loss'].map('${:,.2f}'.format)
    result_df = result_df.transpose()
    list_of_pairs = [result_df[i] for i in result_df]
    return render_template("back_testing.html", pair_list=list_of_pairs)


@app.route('/trade_analysis')
def trade_analysis():
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Generating trading analysis ...")

    select_st = "SELECT printf(\"US$%.2f\", sum(Profit_Loss)) AS Profit, count(Profit_Loss) AS Total_Trades, \
                sum(CASE WHEN Profit_Loss > 0 THEN 1 ELSE 0 END) AS Profit_Trades, \
                sum(CASE WHEN Profit_Loss < 0 THEN 1 ELSE 0 END) AS Loss_Trades FROM StockPairs;"
    result_df = execute_sql_statement(select_st, engine)
    
    'sp500 pnl'
    select_st = "SELECT symbol, date, adjusted_close FROM [GSPC.INDX]"+    \
                " WHERE date >= " + "\"" + str(backtesting_start_date) + "\"" +     \
                " AND date <= " + "\"" + str(backtesting_end_date) + "\"" + ";"
    sp_df = execute_sql_statement(select_st, engine)
    sp_df['ret'] = sp_df['adjusted_close'].pct_change()
    sp_df['cumpnl'] = capital * (1 + sp_df['ret']).cumprod() - capital
    sp_df.index = pd.to_datetime(sp_df.date)

    'Get pnl'
    select_st = 'SELECT Ticker1, Ticker2 FROM stockpairs;'
    pair_df = execute_sql_statement(select_st, engine)
    select_st = 'SELECT * FROM trades;'
    pnl_df = execute_sql_statement(select_st, engine)
    total_pnl = pd.DataFrame(0, columns=["P/L"], index=pnl_df.Date.unique())
    
    for value in pair_df.values:
        pnl = pnl_df.loc[pnl_df.Symbol1==value[0], ["Date","P/L"]]
        pnl.set_index("Date", inplace=True)
        total_pnl = total_pnl.add(pnl)  # adding two dataframe
    
    cumpnl = total_pnl.cumsum()
    maxdraw = MaxDrawdown(cumpnl['P/L'].values)
    result_df["Max_Drawdown"] = maxdraw[-1]
    cumret = cumpnl.pct_change()
    cumret = cumret.replace(np.inf, np.nan)
    cumret = cumret.replace(-np.inf, np.nan)
    result_df["Sharpe"] = np.sqrt(252) * np.nanmean(cumret) / np.nanstd(cumret)
    result_df = result_df.round(2)
    
    print(result_df.to_string(index=False))
    result_df = result_df.transpose()
    trade_results = [result_df[i] for i in result_df]
    
    'plot'
    cumpnl.index = pd.to_datetime(cumpnl.index)
    maxdraw = pd.DataFrame(maxdraw, index=cumpnl.index)
    fig = plt.figure(figsize=(12,7))
    plt.title('Backtesting cumPnL '+str(backtesting_start_date)+' to '+str(backtesting_end_date), 
              fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('PnL (dollars)')
    plt.plot(cumpnl, label='pairs trading pnl')
    plt.plot(maxdraw, label='maximum drawdown')
    plt.plot(sp_df['cumpnl'], label='benchmark(sp500) pnl')
    plt.legend()
    plt.tight_layout()
    fig.savefig('static/plots/backtest_pnl.jpg')
    plt.show()
    return render_template("trade_analysis.html", trade_list=trade_results)


@app.route('/real_trade')
def real_trade():
    global bClientThreadStarted, client_thread
    
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Real trading ...", bClientThreadStarted)

    if bClientThreadStarted == False:
        client_thread.start()
        bClientThreadStarted = True
        print("Client thread starts ...", bClientThreadStarted)
        client_thread.join()  # wait until this thread finishes, then continue main thread

    'real trade analysis'    
    print(" >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Trading analysis ...")
    get_orders(market_period_list[-1])
    stocks_df = pd.DataFrame(price_data, columns=['symbol','date','adjusted_close'])
    stocks_df.adjusted_close = stocks_df.adjusted_close.astype(float)
    total_pnl = pd.Series(0, index=stocks_df.date.unique())
    
    try:
        for stock in record_order_df.index.levels[0]:
            order_df = record_order_df.loc[stock,:]
            stock_df = stocks_df[stocks_df['symbol']==stock]
            stock_df.set_index('date', inplace=True)
            join_df = stock_df.join(order_df)
            join_df.fillna(method='ffill', inplace=True)            
            join_df['pnl'] = (join_df['adjusted_close'] - join_df['Price']) * join_df['Quantity']
            total_pnl = total_pnl.add(join_df.pnl, fill_value=0)  # series + series
    except:
        pass  # if no orders
    
    result_df = pd.DataFrame()
    result_df.loc[0,'Profits'] = sum(total_pnl)
    result_df.loc[0,'Total_Trades'] = len(record_order_df) / 2
    
    cumpnl = total_pnl.cumsum()
    maxdraw = MaxDrawdown(cumpnl.values)
    result_df.loc[0,"Max_Drawdown"] = maxdraw[-1]
    cumret = cumpnl.pct_change()
    cumret = cumret.replace(np.inf, np.nan)
    cumret = cumret.replace(-np.inf, np.nan)
    result_df.loc[0,"Sharpe"] = np.sqrt(30) * np.nanmean(cumret) / np.nanstd(cumret)
    result_df = result_df.round(2)
    
    print(result_df)
    result_df = result_df.transpose()
    trade_results = [result_df[i] for i in result_df]
    
    'sp500 pnl'
    select_st = "SELECT symbol, date, adjusted_close FROM [GSPC.INDX]"+    \
                " WHERE date >= " + "\"" + str(market_period_list[0]) + "\"" +     \
                " AND date <= " + "\"" + str(market_period_list[-1]) + "\"" + ";"
    sp_df = execute_sql_statement(select_st, engine)
    sp_df['ret'] = sp_df['adjusted_close'].pct_change()
    sp_df['cumpnl'] = capital * (1 + sp_df['ret']).cumprod() - capital
    sp_df.index = pd.to_datetime(sp_df.date)
    
    'plot'
    cumpnl.index = pd.to_datetime(cumpnl.index)
    maxdraw = pd.DataFrame(maxdraw, index=cumpnl.index)
    fig = plt.figure(figsize=(12,7))
    plt.title('Trading cumPnL '+str(market_period_list[0])+' to '+str(market_period_list[-1]), 
              fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('PnL (dollars)')
    plt.plot(cumpnl, label='pairs trading pnl')
    plt.plot(maxdraw, label='maximum drawdown')
    plt.plot(sp_df['cumpnl'], label='benchmark(sp500) pnl')
    plt.legend()
    plt.tight_layout()
    fig.savefig('static/plots/trade_pnl.jpg')
    plt.show()

    return render_template("real_trade.html", trade_list=trade_results)

  
    
if(len(sys.argv) > 1) :
    clientID = sys.argv[1]
else:
    clientID = "Yicheng"

HOST = socket.gethostbyname(socket.gethostname())
PORT = 6500
BUFSIZ = 1024
ADDR = (HOST, PORT)

client_socket = socket.socket(AF_INET, SOCK_STREAM)  # create TCP/IP socket
client_socket.connect(ADDR)



if __name__ == "__main__":
    market_period_list = []
    price_data = []
    record_order_df = pd.DataFrame()
    ols_results = {}

    'real trade'
    e = threading.Event()
    q = queue.Queue()
    client_thread = threading.Thread(target=join_trading_network, args=(e,q))
    
    'dashboard'
    bClientThreadStarted = False
    app.run()
