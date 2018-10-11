For each of the data sets, I tried converting them to a time series, similar to what is done for stock market forcasts. I then fed the normalized data through a neural network. 
The solar data network was built on an ANN with 4 layers, using reLu activation. I noticed the loss was not moving very much using an RNN, but after switching to an ANN the loss 
came down faster. The Standard Error, however, continued to explode no matter what architecture I used. I believe this has to do with large outliers in the data. 

The load data was easier to make a time series for. Again, the loss did not minimize that much and the standard error didn't move. The accuracy was never very high no matter 
the architecture I chose. 

I had started making predictions using regular tensorflow and had better results. I would like to invesitage to see if TFLearn is doing something different that
is affect the outcome of the results. I noted many people switching from keras to tflearn experienced similar accuracy issues. 
