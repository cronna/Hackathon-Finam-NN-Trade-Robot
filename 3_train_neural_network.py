import os
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import pytorch_lightning as pl

def load_data(csv_file, symbol="BTCUSDT"):
    df = pd.read_csv(csv_file)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time")
    df["time_idx"] = np.arange(len(df))
    df["target"] = df["close"].astype(float)
    df["group"] = symbol
    return df

if __name__ == "__main__":
    # Загружаем данные для одного тикера, например BTCUSDT
    csv_file = "csv/BTCUSDT.csv"
    data = load_data(csv_file, symbol="BTCUSDT")
    
    # Параметры модели: будем использовать 100 временных шагов для кодировщика и 10 для предсказания.
    # Пример блока для создания датасета с отладочным выводом:
    max_encoder_length = 30  # попробуйте снизить, если недостаточно записей
    max_prediction_length = 10  # попробуйте снизить, если недостаточно записей
    training_cutoff = data["time_idx"].max() - max_prediction_length

    print("Общее количество записей:", len(data))
    print("training_cutoff:", training_cutoff)
    training_data = data[data["time_idx"] <= training_cutoff]
    print("Количество записей после фильтра:", len(training_data))

    training_dataset = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
    )
    
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)
    
    # Создаем модель TFT с указанными гиперпараметрами.
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=QuantileLoss(),
    )
    
    trainer = pl.Trainer(max_epochs=10)   # Если доступен GPU, измените gpus=0 на gpus=1
    trainer.fit(tft, train_dataloader)
    
    os.makedirs("NN_winner", exist_ok=True)
    model_path = "NN_winner/crypto_model_tft.pth"
    tft.save(model_path)
    print("Модель TFT обучена и сохранена по пути:", model_path)