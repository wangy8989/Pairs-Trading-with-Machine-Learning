# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import socket
from threading import Thread
import json
import urllib.request
import sys
import pandas as pd
import random
import sched, time
import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy import MetaData


serverID = "Server1"

startDate = dt.datetime(2019,1,1)  # hours:minute:seconds
endDate = dt.date.today()  # only dates
requestURL = "https://eodhistoricaldata.com/api/eod/"
myEodKey = "5ba84ea974ab42.45160048"

' trading '
engine = create_engine('sqlite:///pairs_trading.db')
engine.execute("PRAGMA foreign_keys = ON")
metadata = MetaData()
metadata.reflect(bind=engine)  # bind to Engine, load all tables


def get_daily_data(symbol='', start=startDate, end=endDate, requestType=requestURL, 
                   apiKey=myEodKey, completeURL=None):
    if not completeURL:
        symbolURL = str(symbol) + '?'
        startURL = "from=" + str(start)
        endURL = "to=" + str(end)
        apiKeyURL = "api_token=" + myEodKey
        completeURL = requestURL + symbolURL + startURL + '&' + endURL + '&' + apiKeyURL + '&period=d&fmt=json'
    print(completeURL)
    
    # if cannot open url
    try:
        with urllib.request.urlopen(completeURL) as req:
            data = json.load(req)
            return data
    except:
        pass 
    

def accept_incoming_connections():
    while True:
        client, client_address = platform_server.accept()
        print("%s:%s has connected." % client_address)
        client_thread = Thread(target=handle_client, args=(client,))
        client_thread.setDaemon(True)
        client_thread.start()

        
def handle_client(client):  
    """Handles a single client connection."""
    global symbols
    price_unit = 0.001
    client_msg = client.recv(buf_size).decode("utf8")
    data = json.loads(client_msg)
    print(data)
    clientID = data["Client"]
    status = data["Status"]
    msg_end_tag = ".$$$$"
    
    if status == "Logon":
        
        if (clientID in clients.values()):
            text = "%s duplicated connection request!" % clientID
            server_msg = "{\"Server\":\"" + serverID + "\", \"Response\":\"" + text + "\", \"Status\":\"Rejected\"}"
            server_msg = "".join((server_msg, msg_end_tag))
            client.send(bytes(server_msg, "utf8"))
            print(text)
            client.close()
            return 
        
        else:
            text = "Welcome %s!" % clientID
            server_msg = "{\"Server\":\"" + serverID + "\", \"Response\":\"" + text + "\", \"Status\":\"Ack\"}"
            server_msg = "".join((server_msg, msg_end_tag))
            client.send(bytes(server_msg, "utf8"))
            clients[client] = clientID
            print (clients[client])
            client_symbols = list(data["Stocks"].split(','))
            symbols.extend(client_symbols)
            symbols = sorted(set(symbols))
            
    try: 
        while True:
            msg = client.recv(buf_size).decode("utf8")
            data = json.loads(msg)
            print(data)
            
            if data["Status"] == "Quit":
                text = "%s left!" % clientID
                server_msg = "{\"Server\":\"" + serverID + "\", \"Response\":\"" + text + "\", \"Status\":\"Done\"}"
                print(server_msg)
            
            elif data["Status"] == "Order Inquiry":
                    if "Symbol" in data and data["Symbol"] != "":
                        server_msg = json.dumps(order_table.loc[order_table['Symbol'].isin(data["Symbol"])].to_json(orient='table'))
            
            elif data["Status"] == "New Order":
                if market_status == "Market Closed":
                    data["Status"] = "Order Reject"
               
                if  ((order_table["Symbol"] == data["Symbol"]) &
                    (order_table["Side"] != data["Side"]) &
                    (abs(order_table["Price"] - float(data["Price"])) < price_unit) &
                    (order_table["Status"] != 'Filled')).any():
                    
                    mask = (order_table["Symbol"] == data["Symbol"]) & \
                            (order_table["Side"] != data["Side"]) & \
                            (abs(order_table["Price"] - float(data["Price"])) < price_unit) & \
                            (order_table["Status"] != 'Filled')
                    order_qty = order_table.loc[(mask.values), 'Qty']
                    
                    if (order_qty.item() == data['Qty']):
                        order_table.loc[(mask.values), 'Qty'] = 0
                        order_table.loc[(mask.values), 'Status'] = 'Filled'
                        data["Status"] = "Fill"
                    elif (order_qty.item() < data['Qty']):
                        data['Qty'] = order_qty.item()  # return your quantity
                        order_table.loc[(mask.values), 'Qty'] = 0
                        order_table.loc[(mask.values), 'Status'] = 'Filled'
                        data["Status"] = "Order Partial Fill"
                    else:
                        order_table.loc[(mask.values), 'Qty'] -= data['Qty']
                        order_table.loc[(mask.values), 'Status'] = 'Partial Filled'
                        data["Status"] = "Order Fill"
                   
                else:
                    if market_status == "Pending Closing":
                        order_table_for_pending_closing = order_table[(order_table["Symbol"] == data["Symbol"]) &
                                                                      (order_table["Side"] != data["Side"])].iloc[[0,-1]]
                        prices = order_table_for_pending_closing["Price"].values
                        
                        if data["Side"] == "Buy":
                            price = float(prices[0])
                            price += 0.01
                        else:
                            price = float(prices[-1])
                            price -= 0.01
                        data["Price"] = str(round(price,2))
                        data["Status"] = "Order Fill"
                    else:
                       data["Status"] = "Order Reject" 
#                print(data)
                server_msg = json.dumps(data)
            
            elif data["Status"] == "User List":
                user_list = str('')
                for clientKey in clients:
                    user_list += clients[clientKey] + str(',')
                server_msg = json.dumps({'User List':user_list})
            
            elif data["Status"] == "Stock List":
                #stock_list = symbols.str.cat(sep=',')
                stock_list = ','.join(symbols)
                server_msg = json.dumps({"Stock List":stock_list})
            
            elif data["Status"] == "Market Status":
                server_msg = json.dumps({"Server":serverID, "Market Status":market_status, "Market Period":market_period})
            
            else:
                text = "Unknown Message from Client"
                server_msg = "{\"Server\":\"" + serverID + "\", \"Response\":\"" + text + "\", \"Status\":\"Unknown Message\"}"
                print(server_msg)
                
            server_msg = "".join((server_msg, msg_end_tag))
            client.send(bytes(server_msg, "utf8"))
            
            if data["Status"] == "Quit":
                client.close()
                del clients[client]
                users = ''
                for clientKey in clients:
                    users += clients[clientKey] + ','
                    print(users)
                return
            
    except KeyboardInterrupt:
        sys.exit(0) 
        
    except json.decoder.JSONDecodeError:
        del clients[client]
        sys.exit(0)
        
clients = {}


def generate_qty(number_of_qty):
    total_qty = 0
    list_of_qty = []
    for index in range(number_of_qty):
        qty = random.randint(1,101)
        list_of_qty.append(qty)
        total_qty += qty
    return (total_qty, list_of_qty)
    
    
def populate_order_table(symbols, start, end):
    price_scale = 0.05
    global order_index, order_table
    order_table.drop(order_table.index, inplace=True)
    
    for symbol in symbols:
        stock = get_daily_data(symbol, start, end)
        
        for stock_data in stock:
            (total_qty, list_of_qty) = generate_qty(int((float(stock_data['high'])-float(stock_data['low']))/price_scale))
            buy_price = float(stock_data['low']);
            sell_price = float(stock_data['high'])
            daily_volume = float(stock_data['volume'])
            
            for index in range(0, len(list_of_qty)-1, 2):
                order_index += 1
                order_table.loc[order_index] = [order_index, symbol, 'Buy', buy_price, int((list_of_qty[index]/total_qty)*daily_volume), 'New']
                buy_price += 0.05
                order_index += 1
                order_table.loc[order_index] = [order_index, symbol, 'Sell', sell_price, int((list_of_qty[index+1]/total_qty)*daily_volume), 'New']
                sell_price -= 0.05
            
    print(order_table)
    print(market_status, market_period)
    
    
'''
(1) Server will provide consolidated books for 30 trading days, 
    (a) simulated from market data starting from 1/2/2019.
    (b) Each simulated trading date has one book, with buy orders and sell orders 
        simulated from the high and low price from the day, with daily volume randomly 
        distributed cross all price points.
    (c) Each simulated trading date starts with a new book simulated from corresponding 
        daily historical data
'''
def create_market_interest(index):
    global market_period, symbols
    
    market_periods = pd.bdate_range('2019-01-02', '2019-04-01').strftime("%Y-%m-%d").tolist()
    
    # in order
    startDate = market_periods[index]
    endDate = market_periods[index]
    
    if len(order_table) == 0 or (market_status != "Market Closed" and market_status != "Pending Closing"):
        market_period = startDate
        populate_order_table(symbols, startDate, endDate)
        print(market_status, "Creating market interest")
    else:
        print(market_status, "No new market interest")

'''
(2) Each simulated trading day lasts 30 seconds, 
    following by 5 seconds of pending closing phase 
    and 5 seconds of market closed phase before market reopen
'''
def update_market_status(status, day):
    global market_status
    global order_index
    global order_table
    
    market_status = status
    create_market_interest(day)
    
    market_status = 'Open'
    print(market_status)
    time.sleep(30)
    
    market_status = 'Pending Closing'
    print(market_status)
    time.sleep(5)
    
    market_status = 'Market Closed'
    print(market_status)
    
    order_table.fillna(0)
    order_index = 0
    time.sleep(5)

'''
(3) There are 5 phases of market: 
    (a) Not Open, start
    (b) Pending Open, 
    (c) Open,  30
    (d) Pending Close, 5
    (e) Market Closed 5
'''
def set_market_status(scheduler, time_in_seconds):
    value = dt.datetime.fromtimestamp(time_in_seconds)
    print(value.strftime('%Y-%m-%d %H:%M:%S'))
    
    # 40s for one day
    for day in range(total_market_days):
        scheduler.enter(40*day+1,1, update_market_status, argument=('Pending Open', day))
    scheduler.run()
    
    
port = 6500
buf_size = 1024
platform_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(socket.gethostname())
platform_server.bind((socket.gethostname(), port))

if __name__ == "__main__":
    
    market_status = "Not Open"
    market_period = "2019-01-01"
    order_index = 0
    total_market_days = 30
    
    symbols = []
    order_table_columns = ['OrderIndex', 'Symbol', 'Side', 'Price', 'Qty', 'Status']
    order_table = pd.DataFrame(columns=order_table_columns)
    order_table = order_table.fillna(0)
            
    platform_server.listen(1)
    print("Waiting for client requests")
    time.sleep(80)  # wait for backtesting to finish
    
    try:
        scheduler = sched.scheduler(time.time, time.sleep)
        current_time_in_seconds = time.time()
        scheduler_thread = Thread(target=set_market_status, args=(scheduler, current_time_in_seconds))
        scheduler_thread.setDaemon(True)
        
        server_thread = Thread(target=accept_incoming_connections)
        server_thread.setDaemon(True)
        
        server_thread.start()
        scheduler_thread.start()
        
        scheduler_thread.join()  # wait until scheduler finished
        server_thread.join()  # server finish after scheduler finished
        
    except (KeyboardInterrupt, SystemExit):
        platform_server.close() 
        sys.exit(0)
        