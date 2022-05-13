# Pairs-Trading-with-Machine-Learning

Notebook file: implement strategy on SP500
1. Implemented PCA and DBSCAN clustering to group SP500 stocks based on similar factor loadings  
2. Identified pairs within clusters to implement dollar neutral Bollinger Band pairs trading strategy  
3. Constructed portfolio with pairs equally weighted  
Result: This portfolio achieved has a 2.5 Sharpe ratio and 25% annual return in 2018.

## Pairs-Trading-with-Machine-Learning-on-Distributed-Platform
PY files: implement strategy on Russell3000  
It also implements a distributed Python platform that can be used to test quantitative models for trading financial instruments in a network setting under client/server infrastructure. 

* Codes are in "platform_server.py" and "platform_client.py"; 
* database is "pairs_trading.db"; 
* templates for flask are in "templates" folder; 
* "static" folder has PnL plots.

Instructions:
1. Run "platform_server.py";
2. Open another console and run "platform_client.py";
3. Open web browser and go to "http://127.0.0.1:5000/", then home page will show;
4. Click "Stock Pairs" -> "Building Model" -> "Back Testing" -> "Trading Analysis" -> "Real Trading" in order;
video instructions are in "video_flask" for web browser and "video_program" for running program.

Please refer to "Capstone_Final_Yicheng_Wang.rar" for complete files and data.
