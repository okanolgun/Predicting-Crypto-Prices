import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import mplfinance
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

import yfinance as yf

data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)


# preparing data
# print(data.head())

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60
# we'll predict the price with the last 60 days

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# we are going to use the time period from x - predictions days
# up until x to predict the x, the actual x

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
