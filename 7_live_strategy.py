import time
import json
import torch
from binance.client import Client
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from my_config.trade_config import trade_config

# Загрузка параметров датасета
with open("NN_winner/dataset_params.json", "r") as f:
    dataset_params = json.load(f)

# Загрузка модели
model = TemporalFusionTransformer.load_from_checkpoint(
    "NN_winner/crypto_model_tft.ckpt",
    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
model.eval()

# Конфигурация API
client = Client("uTCEuEO73fCXtNKgNFQXdOtQmWLDU7DvnrfW15Cj787PV7c9juQM4LyYmAuz4a7s", "mqLuFRdapde2kHizPnI67trMQ1hjuc4N42bzshqHm4ebi2NjsgT4Dn10fGHyYLTu")

crypto_symbols = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT"
}

def get_recent_data(symbol):
    klines = client.get_historical_klines(
        symbol, 
        Client.KLINE_INTERVAL_1HOUR, 
        "200 hours ago UTC"
    )
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.sort_values("open_time")
    df["time_idx"] = np.arange(len(df))
    df["target"] = df["close"].astype(float)
    df["group"] = symbol
    return df

def predict_action(symbol):
    df = get_recent_data(symbol)
    
    # Рассчитываем требуемую длину данных
    required_length = dataset_params["max_encoder_length"] + dataset_params["max_prediction_length"]
    if len(df) < required_length:
        print(f"Недостаточно данных для {symbol} (нужно {required_length}, есть {len(df)})")
        return
    
    data_for_pred = df.iloc[-required_length:].copy()
    
    # Создаем dataset с сохраненными параметрами
    prediction_dataset = TimeSeriesDataSet(
        data_for_pred,
        time_idx="time_idx",
        target=dataset_params["target"],
        group_ids=dataset_params["group_ids"],
        max_encoder_length=dataset_params["max_encoder_length"],
        max_prediction_length=dataset_params["max_prediction_length"],
        time_varying_known_reals=dataset_params["time_varying_known_reals"],
        time_varying_unknown_reals=dataset_params["time_varying_unknown_reals"],
    )
    
    # Прогнозирование
    test_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    predictions = model.predict(test_dataloader)
    
    # Логика принятия решения
    predicted_mean = predictions.numpy().mean()
    last_price = data_for_pred["target"].iloc[-1]
    decision = "BUY" if predicted_mean > last_price else "SELL"
    print(f"{symbol}: Прогноз = {predicted_mean:.2f}, Текущая цена = {last_price:.2f} → {decision}")

if __name__ == "__main__":
    while True:
        for crypto in trade_config.portfolio:
            symbol = crypto_symbols.get(crypto)
            if symbol:
                try:
                    predict_action(symbol)
                except Exception as e:
                    print(f"Ошибка для {symbol}: {str(e)}")
        time.sleep(3600)