import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import torch

# Загрузка данных для тестирования
csv_file = "csv/BTCUSDT.csv"
df = pd.read_csv(csv_file)
df["open_time"] = pd.to_datetime(df["open_time"])
df = df.sort_values("open_time")
df["time_idx"] = np.arange(len(df))
df["target"] = df["close"].astype(float)
df["group"] = "BTCUSDT"

max_encoder_length = 30
max_prediction_length = 10

# Подготовка данных для предсказания
data_for_pred = df.iloc[-(max_encoder_length + max_prediction_length):].copy()

# Создаем dataset (должен быть идентичный тому, что использовался при обучении)
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

# Загрузка модели
model = TemporalFusionTransformer.from_dataset(
    prediction_dataset,  # Должен быть тот же dataset, что использовался для обучения
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
)

model.load_state_dict(torch.load("NN_winner/crypto_model_tft.pth"))
model.eval()
predictions = model.predict(test_dataloader)
print("Предсказания модели TFT для BTCUSDT:")
print(predictions)