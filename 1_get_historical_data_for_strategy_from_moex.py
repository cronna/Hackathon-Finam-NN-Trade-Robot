from binance.client import Client
import pandas as pd
import os

# Укажите свои API-ключи Binance
API_KEY = "uTCEuEO73fCXtNKgNFQXdOtQmWLDU7DvnrfW15Cj787PV7c9juQM4LyYmAuz4a7s"
API_SECRET = "mqLuFRdapde2kHizPnI67trMQ1hjuc4N42bzshqHm4ebi2NjsgT4Dn10fGHyYLTu"
client = Client(api_key=API_KEY, api_secret=API_SECRET, testnet=True)

def get_crypto_history(symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    os.makedirs("csv", exist_ok=True)
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    data = pd.DataFrame(klines, columns=columns)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.to_csv(f"csv/{symbol}.csv", index=False)
    print(f"Сохранены исторические данные для {symbol}")

if __name__ == "__main__":
    # Пример для нескольких криптовалют
    cryptos = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "LTC": "LTCUSDT", "BCH": "BCHUSDT"}
    for crypto, symbol in cryptos.items():
        print(f"Получение данных для {crypto}")
        get_crypto_history(symbol, Client.KLINE_INTERVAL_1HOUR, "1 Jan 2023")