"""
    В этом коде мы асинхронно получаем исторические данные с MOEX
    и сохраняем их в CSV файлы. Т.к. получаем их бесплатно, то
    есть задержка в полученных данных на 15 минут.
    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе перепишет результаты

import asyncio
import aiohttp
import aiomoex
import functions
import os
import pandas as pd
from datetime import datetime, timedelta
from my_config.trade_config import Config  # Файл конфигурации торгового робота


from binance.client import Client
import pandas as pd
import os

# Replace with your actual Binance API credentials
API_KEY = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"
client = Client(api_key=API_KEY, api_secret=API_SECRET)

def get_crypto_history(symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    os.makedirs("csv", exist_ok=True)
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades', 
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    data = pd.DataFrame(klines, columns=columns)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.to_csv(f"csv/{symbol}.csv", index=False)
    print(f"Saved historical data for {symbol}")

if __name__ == "__main__":
    cryptos = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "LTC": "LTCUSDT", "BCH": "BCHUSDT"}
    for crypto, symbol in cryptos.items():
        print(f"Fetching historical data for {crypto}")
        get_crypto_history(symbol, Client.KLINE_INTERVAL_1HOUR, "1 Jan 2023")


