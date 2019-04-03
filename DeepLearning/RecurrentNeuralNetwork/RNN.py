# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
'''
These warnings filter are to unclutter output and easy-to-read purposes only!
Remember to lookup replacements. Uncomment for info
Reminder: some functions in sckitlearn have been/will be depreceated,

'''
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

### ----------------------- Data-Preprocessing ----------------------------- ###
dataset_train = pd.read_csv('../_data/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# scale features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

### ------------------------------ Building -------------------------------- ###
# initialising the RNN
regressor = Sequential()

# first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,
                return_sequences = True,
                input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,
                return_sequences = True))
regressor.add(Dropout(0.2))

# third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,
                return_sequences = True))
regressor.add(Dropout(0.2))

# fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



### ----------------------- Result & Visualisation ------------------------- ###

# uploading the real stock price of 2017
dataset_test = pd.read_csv('../_data/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# comparing the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red',
        label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue',
        label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
