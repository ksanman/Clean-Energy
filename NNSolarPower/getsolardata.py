import pandas as pd
import math
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import time
from sklearn.preprocessing import MinMaxScaler

class SolarData:

    def __init__(self, *args, **kwargs):
        self.filepath = 'D:/Xaber/Documents/School/Fall 2018/CS5890/Class Data/NNSolarPower/hourly-weather-dataset_chronological-order.csv'
        self.dateparse = lambda d, t: self.get_date_time(d,t)

    def get_time(self, stime):
        return str(stime) + ':00'

    def get_date_time(self, date, time):
        return datetime.strptime(date + ' ' + self.get_time(time), "%m/%d/%Y %H:%M")

    def derive_nth_day_feature(self,df, feature, N):  
        rows = df.shape[0]
        nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
        col_name = "{}_{}".format(feature, N)
        df[col_name] = nth_prior_measurements

    def derive_features(self, raw_df,numberOfDays):
        features = list(raw_df)
        for feature in features:
            if feature != 'Date_Hour':
              for N in range(1, numberOfDays):
                    self.derive_nth_day_feature(raw_df, feature, N)

    def remove_nan(self, raw_df, value):
        for feature in list(raw_df):  
            # create a boolean array of values representing nans
            missing_vals = pd.isnull(raw_df[feature])
            raw_df[feature][missing_vals] = value

    def get_train_and_test_data(self, value=0, numberOfDays=1):     
        raw_df = pd.read_csv(self.filepath, parse_dates= [[0, 1]], index_col=0, date_parser=self.dateparse)
        print(raw_df.head())
        self.derive_features(raw_df,numberOfDays)
        print(raw_df.head())
        self.remove_nan(raw_df, value)
        raw_df.info()

        #Seperate data into X and Y
        Y = raw_df['Solar energy'].values
        X = raw_df.drop('Solar energy',1).values

        #Scale the data:
        #X = X.reshape(len(X), 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(X)
        X = scaler.transform(X)
        #Y = Y.reshape(len(X), 1)
        #scaler = MinMaxScaler()
        #scaler = scaler.fit(Y)
        #Y = scaler.transform(Y)

        #Break into training, testing, and validation sets
        percentage = int(np.floor(X.shape[0]*.80))
        train_x, tmp_x = X[:percentage], X[percentage:]
        train_y, tmp_y = Y[:percentage], Y[percentage:]
        half = int(np.floor(tmp_x.shape[0]*.50))
        test_x, valid_x = tmp_x[:-half], tmp_x[-half:]
        test_y, valid_y = tmp_y[:-half], tmp_y[-half:]
        train_y = train_y.reshape(train_y.shape[0], 1)
        test_y = test_y.reshape(test_y.shape[0], 1)
        valid_y = valid_y.reshape(valid_y.shape[0], 1)
        return train_x, train_y, test_x, test_y, valid_x, valid_y