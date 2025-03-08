import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import torch

# Загрузка данных для тестирования (например, для BTCUSDT)
csv_file = "csv/BTCUSDT.csv"
df = pd.read_csv(csv_file)
df["open_time"] = pd.to_datetime(df["open_time"])
df = df.sort_values("open_time")
df["time_idx"] = np.arange(len(df))
df["target"] = df["close"].astype(float)
df["group"] = "BTCUSDT"

max_encoder_length = 100
max_prediction_length = 10

# Используем последние (encoder + prediction) строк
data_for_pred = df.iloc[-(max_encoder_length + max_prediction_length):].copy()

prediction_dataset = TimeSeriesDataSet(
    data_for_pred,
    time_idx="time_idx",
    target="target",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"],
)

test_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# Загрузка модели TFT
model = TemporalFusionTransformer.load_from_checkpoint("NN_winner/crypto_model_tft.pth")

predictions = model.predict(test_dataloader)
print("Предсказания модели TFT для BTCUSDT:")
print(predictions)
