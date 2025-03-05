"""
    В этом коде реализована live стратегия, мы однократно асинхронно
    получаем исторические данные с MOEX, и считаем что следующие асинхронно
    полученные исторические данные приходят нам в live режиме.
    Т.к. получаем их бесплатно, то есть задержка в полученных данных на 15 минут.

    Используем нейросеть для прогноза о вхождении в сделку:
    - нейросеть выбираем на шаге №3, по результатам loss, accuracy, val_loss, val_accuracy
    - открываем позицию по рынку, как только получаем сигнал от нейросети с классом 1 - на покупку 1 лотом
    - без стоп-лосса, т.к. закрывать позицию будем на следующем +1 баре старшего таймфрейма по рынку

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

import asyncio
import os.path

import aiohttp
import aiomoex
import logging
import functions
import functions_nn
import pandas as pd
import numpy as np
from aiohttp import ClientSession
from datetime import datetime, timedelta
from typing import List, Optional

from FinamPy import FinamPy  # Коннект к Финам API - для выставления заявок на покупку/продажу
from FinamPy.proto.tradeapi.v1.common_pb2 import BUY_SELL_BUY, BUY_SELL_SELL

from my_config.Config import Config as ConfigAPI  # Файл конфигурации Финам API
from my_config.trade_config import Config  # Файл конфигурации торгового робота

from keras.models import load_model
from keras.utils.image_utils import img_to_array


import time
from binance.client import Client
import numpy as np
import tensorflow as tf
import pandas as pd
from my_config import trade_config  # Imports training_NN and portfolio

# Replace with your Binance API credentials
API_KEY = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"
client = Client(API_KEY, API_SECRET)

model = tf.keras.models.load_model("NN_winner/crypto_model.hdf5", compile=False)

def get_recent_data(symbol):
    # Fetch the most recent 100 hourly klines for the provided symbol.
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "100 hours ago UTC")
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    # Convert the 'close' column to floats.
    df['close'] = df['close'].astype(float)
    # For demonstration, we create a feature matrix by repeating the 'close' price to form 10 features.
    data = np.tile(df['close'].values.reshape(-1, 1), (1, 10))
    return data

def trade_decision(symbol):
    data = get_recent_data(symbol)
    if data.shape[0] < 100:
        print(f"Not enough data for {symbol}")
        return
    input_data = np.expand_dims(data[-100:], axis=0)  # Shape: (1, 100, 10)
    prediction = model.predict(input_data)
    decision = "BUY" if prediction[0][0] > 0.5 else "SELL"
    print(f"Crypto: {symbol}, Prediction: {prediction[0][0]:.3f}, Decision: {decision}")
    # Here you can integrate actual trade execution via the Binance API.

if __name__ == "__main__":
    crypto_symbols = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "LTC": "LTCUSDT", "BCH": "BCHUSDT"}
    while True:
        for crypto in trade_config.portfolio:
            symbol = crypto_symbols.get(crypto)
            if symbol:
                trade_decision(symbol)
        # Sleep for 1 hour before the next decision cycle; adjust for testing if needed.
        time.sleep(3600)