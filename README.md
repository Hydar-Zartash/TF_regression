# TF_regression
Stock performance predicting using tensorflow and yfinance

This project attempts to extract the maximum useful information from a stocks daily market data, 
by extracting as many common technical analysis indicators as it can and using those as inputs to 
a 3 layer Deep neural network. The networks task is to train itself from scratch on the data, then 
predict whether it will grow by X% in the next month from a given date. This ttask is meaningless, but
so is technical analysis for stock picking, so I thought it would be a nice excercse. 
The output layer uses a sigmoid function to try to approximate a boolean true or false as our output.  
