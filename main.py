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
print(data.head())
