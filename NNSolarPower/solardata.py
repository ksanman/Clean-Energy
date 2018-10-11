import getsolardata

#Import the data
train_x, train_y, test_x, test_y, valid_x, valid_y = getsolardata.SolarData().get_train_and_test_data(0,2)

#shape and scale the data


import tflearn as tf
#train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
#test_x =test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
#valid_x = valid_x.reshape(valid_x.shape[0], 1, valid_x.shape[1])
input = tf.input_data(shape=(None, train_x.shape[1]))
emb = tf.embedding(input, input_dim=train_x.size, output_dim=50)
lstm = tf.lstm(emb, 50, activation='linear', dropout=0.8)
#net = tf.fully_connected(input, 34)
net = tf.fully_connected(lstm, 1)
regression = tf.regression(net, optimizer='adam', learning_rate=0.01, loss='mean_square', metric='R2')
model = tf.DNN(regression)

testevaluations = []
trainevaluations = []
for i in range(2):
    model.fit(train_x, train_y, n_epoch=1, batch_size=20, show_metric=True)
    trainevaluations.append(model.evaluate(train_x, train_y))
    testevaluations.append(model.evaluate(test_x, test_y))

import matplotlib.pylab as plt
#plt.plot(range(0,20),trainevaluations)
#plt.plot(range(0,20),testevaluations)
#plt.show()