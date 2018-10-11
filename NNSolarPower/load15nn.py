import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import tflearn as tf
from datetime import datetime
from datetime import timedelta
from datetime import time

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def GetTime(stime):
    t = str(stime)

    if len(t) is 2:
        t = '00:' + t
    elif len(t) is 3:
        t = '0' + t[0] + ':' + t[1] + t[2]
    else:
        h = t[:2]
        if h == '24':
            return '00:00'
        t = t[:2] + ':' + t[2:]
    return t

def GetDateTime(date, time):
    t = GetTime(time)
    return datetime.strptime(date + ' ' + t, "%m/%d/%Y %H:%M")

#Import the data
filepath = 'D:/Xaber/Documents/School/Fall 2018/CS5890/Class Data/NNSolarPower/LoadProfile15Mins.csv'
dateparse = lambda d, t: GetDateTime(d,t)
data = pd.read_csv(filepath, parse_dates= [[0, 1]], index_col=0 ,date_parser=dateparse)

# Create a time series
ts = data['Customer 15 minute']

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
data = data.values
#Preprocess data
diff_values = difference(data, 1)
supervised = timeseries_to_supervised(diff_values, 1)                           

trainingdata = supervised.values
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n - 1
data_train = trainingdata[np.arange(train_start, train_end), :]
data_test = trainingdata[np.arange(test_start, test_end), :]

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
Y_train = Y_train.reshape(Y_train.shape[0],1)
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]
Y_test = Y_test.reshape(Y_test.shape[0],1)

#Create the modal.
input = tf.input_data(shape=(None, p))
hidden1 = tf.fully_connected(input, 3, activation='relu')
hidden2 = tf.fully_connected(hidden1, 1, activation='relu')
regression = tf.regression(hidden2, optimizer='adam', learning_rate=0.001, loss='mean_square', metric='R2')

# Define model
#model = tf.DNN(regression)
model = tf.Generator.SequenceGenerator(regression);
model.fit(X_train, Y_train, n_epoch=20, batch_size=20, shuffle=False, show_metric=True)
print('Accuracy: ', model.evaluate(X_test, Y_test))