# Pairs-Trading-with-Machine-Learning

Notebook file: implement strategy on Russell3000  

It implements the Pairs Trading strategy with Machine Learning to find the most profitable portfolio. 
The idea is based on the stocks that share loadings to common factors in the past should be related in the future.
We used Russell 3000 as our project data from 2010 to 2018 from Bloomberg. 
The information we retrieved contains daily prices of stocks, Global Industry Classification Standard (GICS), analyst rating, market to book value, return on asset, debt to asset, EPS, and market cap.  

Result: the best model with tuned hyperparameters achieved Sharpe ratio 1.55.

## Pairs-Trading-with-Machine-Learning-on-Distributed-Python-Platform

This project implements a distributed Python platform that can be used to test quantitative models for trading financial instruments in a network setting under client/server infrastructure. Normally, we backtest locally using past historical data to check the performance of our trading strategies. The performance result, in this case, is usually an illusion of what the actual performance is in real-time trading. We also show in this paper this conclusion by showing that our quantitative trading model performs much worse in the simulated trading than that in backtesting environment. Therefore, we build this Python platform not only for implementing trading strategies and backtesting them historically but also for simulating trades similar to what is in real market, acting as another control before real-time trading.

Strategy:
1. Implemented PCA and DBSCAN clustering to group SP500 stocks based on similar factor loadings  
2. Identified pairs within clusters to implement dollar neutral Bollinger Band pairs trading strategy  
3. Constructed portfolio with pairs equally weighted  

Result: This portfolio achieved has a 2.5 Sharpe ratio and 25% annual return in 2018.  

* Codes are in "platform_server.py" and "platform_client.py"; 
* database is "pairs_trading.db"; 
* templates for flask are in "templates" folder; 
* "static" folder has PnL plots.

**Download "Capstone_Final_Yicheng_Wang.rar" if you want to run the project (with codes, data and video instruction).**

Instructions:
1. In "Program" folder, run "platform_server.py";
2. Open another console and run "platform_client.py";
3. Open web browser and go to "http://127.0.0.1:5000/", then home page will show;
4. Click "Stock Pairs" -> "Building Model" -> "Back Testing" -> "Trading Analysis" -> "Real Trading" in order;
video instructions are in "video_flask" for web browser and "video_program" for running program.
