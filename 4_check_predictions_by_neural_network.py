"""
    В этом коде реализована проверка предсказаний нейросетью классов картинок.

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

import os
import functions
import numpy as np
from PIL import Image

from keras.models import load_model
from keras.utils.image_utils import img_to_array

from my_config.trade_config import Config  # Файл конфигурации торгового робота

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    model = tf.keras.models.load_model("NN_winner/crypto_model.hdf5", compile=False)
    # Create dummy sample data: 10 samples, each of shape (100, 10)
    X_sample = np.random.rand(10, 100, 10)
    predictions = model.predict(X_sample)
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: Prediction probability: {pred[0]:.3f}")
