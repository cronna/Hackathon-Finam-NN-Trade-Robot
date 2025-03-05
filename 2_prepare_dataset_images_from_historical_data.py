"""
    В этом коде мы подготавливаем данные для обучения нейросети.
    Генерируем картинки по следующему алгоритму:
    1. берем картинку графика цены закрытия + SMA1 + SMA2 за определенный
    интервал на таймфрейме_0
    2. если закрытие на старшем таймфрейме_1 > закрытия предыдущей свечи старшего
    таймфрейма_1, то для этой картинки назначаем класс 1, иначе класс 0
    P.S. SMA1, SMA2 - скользящие средние
    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе перепишет результаты

import pandas as pd
import matplotlib.pyplot as plt
import os

def prepare_chart_image(csv_file, output_dir, crypto, label):
    data = pd.read_csv(csv_file)
    data['open_time'] = pd.to_datetime(data['open_time'])
    data = data.sort_values(by='open_time')
    # Calculate moving averages
    data['MA_short'] = data['close'].astype(float).rolling(window=10).mean()
    data['MA_long'] = data['close'].astype(float).rolling(window=30).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['open_time'], data['close'].astype(float), label='Close Price', color='blue')
    plt.plot(data['open_time'], data['MA_short'], label='MA 10', color='green')
    plt.plot(data['open_time'], data['MA_long'], label='MA 30', color='red')
    plt.title(f"{crypto} Price Chart - Label {label}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    image_file = os.path.join(output_dir, f"{crypto}_{label}.png")
    plt.savefig(image_file)
    plt.close()
    print(f"Saved image for {crypto} as {image_file}")

if __name__ == "__main__":
    csv_dir = "csv"
    output_dir = "NN/training_dataset_M1"
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            csv_file = os.path.join(csv_dir, file)
            crypto = file.split('.')[0]
            data = pd.read_csv(csv_file)
            # Determine label: if the last close is higher than the previous, label "1", else "0"
            if len(data) > 1:
                label = 1 if float(data['close'].iloc[-1]) > float(data['close'].iloc[-2]) else 0
            else:
                label = 0
            prepare_chart_image(csv_file, output_dir, crypto, label)