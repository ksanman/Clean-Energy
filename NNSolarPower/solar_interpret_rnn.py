import numpy as np
import math
import matplotlib as mpl
from matplotlib.image import imread
from random import randint
import pandas as pd
from datetime import datetime
import tflearn as tf

mpl.use('Agg')
import matplotlib.pyplot as plt

#Set y values of data to lie between 0 and 1
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

#Import and pre-process data for future applications
def import_data(train_dataframe, dev_dataframe, test_dataframe):
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')

    end = 11
    count = end + 1

    #Include all 12 initial factors (Year ; Month ; Hour ; Day ; Cloud Coverage ; Visibility ; Temperature ; Dew Point ;
    #Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter
    max_test = np.max(dataset[:,end])
    min_test = np.min(dataset[:,end])
    scale_factor = max_test - min_test
    max = np.empty(count)
    min = np.empty(count)

    #Create training dataset
    for i in range(0,count):
        min[i] = np.amin(dataset[:,i],axis = 0)
        max[i] = np.amax(dataset[:,i],axis = 0)
        dataset[:,i] = normalize_data(dataset[:, i], min[i], max[i])

    #Shuffle the data
    from random import shuffle
    shuffle(dataset)

    train_data = dataset[:,0:end]
    train_labels = dataset[:,end]

    # Create dev dataset
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, count):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    dev_data = dataset[:,0:end]
    dev_labels = dataset[:,end]

    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, count):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    test_data = dataset[:, 0:end]
    test_labels = dataset[:, end]

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor

#boolean to define is the X dataset needs to be reshaped.
shapeX = False

#Construct and return Tflearn RNN model
def build_model(lr, layers):
    net = tf.input_data(shape=[None,11])
    for l in layers:
        net = tf.layers.fully_connected(net, l, activation='relu')

    net = tf.layers.fully_connected(net, 1, activation='relu')
    regression = tf.regression(net, optimizer='adam', learning_rate=lr, loss='mean_square')
    model = tf.DNN(regression, clip_gradients=3, tensorboard_verbose=3, tensorboard_dir='logs/1/trainsess.graph')
    return model

#Save output predictions for graphing and inspection
def write_to_csv(prediction, filename):
    print("Writing to CSV...")
    with open(filename, 'w') as file:
        for i in range(prediction.shape[0]):
            file.write("%.5f" % prediction[i][0][0])
            file.write('\n')
    print("...finished!")

#Return MSE error values of all three data sets based on a single model
def evaluate(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, scale_factor):
    scores = model.evaluate(X_train, Y_train, verbose = 0)#* scale_factor #* scale_factor
    print("train: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_dev, Y_dev, verbose = 0)#) * scale_factor #* scale_factor
    print("dev: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_test, Y_test, verbose = 0)# * scale_factor #* scale_factor
    print("test: ", model.metrics_names, ": ", scores)

#Calculate MSE between two arrays of values
def mse(predicted, observed):
    return np.sum(np.multiply((predicted - observed),(predicted - observed)))/predicted.shape[0]

def get_time(stime):
    return str(stime) + ':00'

def get_date_time(date, time):
    return datetime.strptime(date + ' ' + get_time(time), "%m/%d/%Y %H:%M")

#SCRIPT BEGINS ##########################################################################################

plt.switch_backend('tkAgg')
filepath = 'D:/Xaber/Documents/School/Fall 2018/CS5890/Class Data/NNSolarPower/hourly-weather-dataset_chronological-order.csv'
#dateparse = lambda d, t: get_date_time(d,t)
#df = pd.read_csv(filepath, parse_dates= [[0, 1]], index_col=0, date_parser=dateparse)
df = pd.read_csv(filepath)

#Get the day, month, and time columns set. 
ndf = df.copy()
ndf['Month'] = pd.DatetimeIndex(df['Date']).month
ndf['Day'] = pd.DatetimeIndex(df['Date']).day
ndf = ndf[['Date','Month', 'Day','Hour', 'Cloud coverage', 'Visibility', 'Temperature', 'Dew point', 'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter', 'Solar energy']]
df = ndf.drop('Date', axis=1)
headers = list(df)


data = df.values
data_set_divide = int(np.floor(data.shape[0]*.60))
train_d, remainder = data[:data_set_divide], data[data_set_divide:]
dev_d = remainder[:int(np.floor(remainder.shape[0] * .50))]
test_d = remainder[int(np.floor(remainder.shape[0] * .50)):]

train_dataframe = pd.DataFrame(train_d)
dev_dataframe = pd.DataFrame(dev_d)
test_dataframe = pd.DataFrame(test_d)

train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor = import_data(train_dataframe, dev_dataframe, test_dataframe)

if shapeX is True:
    X_train = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    X_dev = np.reshape(dev_data, (dev_data.shape[0], 1, dev_data.shape[1]))
    X_test = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
else:
    X_train = train_data
    X_dev = dev_data
    X_test = test_data

Y_train = train_labels.reshape(train_labels.shape[0], 1)
Y_dev = dev_labels.reshape(dev_labels.shape[0], 1)
Y_test = test_labels.reshape(test_labels.shape[0], 1)


models =[[100,200,100,50,10],[150,150,100,50,10],[200,150,100,50,10],[100,50,10],[100,12],[24,24,10]]
learningrates = [0.0001,0.001,0.0005,0.0003,0.005,0.003]

model_fit_epochs = 100

for m in models:
    for lr in learningrates:
        from tensorflow import reset_default_graph
        reset_default_graph()
        model = build_model(lr,m)
        model.fit(X_train, Y_train, batch_size=20, n_epoch=model_fit_epochs, show_metric=True, shuffle=True, validation_set=(X_dev, Y_dev))
        trainset_predicted = model.predict(X_train)
        devset_predicted = model.predict(X_dev)
        testset_predicted = model.predict(X_test)

        unscaled_predict = trainset_predicted * scale_factor
        y_scale = Y_train * scale_factor
        dev_pred = devset_predicted * scale_factor
        dev_y_scale = Y_dev * scale_factor
        valid_pred = testset_predicted * scale_factor
        valid_truth = Y_test * scale_factor

        from sklearn.metrics import mean_squared_error
        rms = math.sqrt(mean_squared_error(y_scale, unscaled_predict))
        rms2 = math.sqrt(mean_squared_error(dev_y_scale, dev_pred))
        rms3 = math.sqrt(mean_squared_error(valid_truth, valid_pred))

        num_lay = len(m)
        l = ''.join(str(l) for l in m)
        msg = f'Training Results for model with {num_lay} layers of {l} neurons and a learning rate of {lr} \n\n Training Root mean square error: ' + str(rms)\
           +'\n\n Testing Root mean square error: ' + str(rms2) + '\n\n Valid Root mean square error: ' +str(rms3)

        folder_name = f'model_{num_lay}_{l}_{lr}'
        import os
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        f = open(folder_name +'/nn_solar_stats.txt', 'a+')
        f.write(msg)
        f.close()

        #print('Training Root mean square error: ',rms)
        #print('Testing Root mean square error: ',rms2)
        #print('Valid Root mean square error: ',rms3)

        plt.plot(unscaled_predict, label='predicted')
        plt.plot(y_scale, label='truth')
        plt.title("Predicted vs. Training Solar Power")
        plt.ylabel("Solar power")
        plt.xlabel("Time")
        plt.legend()
        fig = plt.figure()
        fig.savefig(folder_name + '/Training_Figure.png')

        plt.plot(dev_pred, label='predicted')
        plt.plot(dev_y_scale, label='truth')
        plt.title("Predicted vs. Testing Solar Power")
        plt.ylabel("Solar power")
        plt.xlabel("Time")
        fig = plt.figure()
        fig.savefig(folder_name + '/Test_Figure.png')

        plt.plot(valid_pred, label='predicted')
        plt.plot(valid_truth, label='truth')
        plt.title("Predicted vs. validation Solar Power")
        plt.ylabel("Solar power")
        plt.xlabel("Time")
        fig = plt.figure()
        fig.savefig(folder_name + '/Validation_Figure.png')