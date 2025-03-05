
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, Softmax, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
import os

def attention_block(inputs):
    # Compute attention scores and weights, then derive a weighted sum of LSTM outputs.
    attention_scores = Dense(1, activation='tanh')(inputs)  # (batch, timesteps, 1)
    attention_scores = tf.keras.layers.Flatten()(attention_scores)  # (batch, timesteps)
    attention_weights = Softmax()(attention_scores)  # (batch, timesteps)
    attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch, timesteps, 1)
    attended_output = Multiply()([inputs, attention_weights])
    # Sum over the time dimension
    output = Lambda(lambda x: K.sum(x, axis=1))(attended_output)
    return output

def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)  # e.g., (100, 10)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attended = attention_block(lstm_out)
    output = Dense(1, activation='sigmoid')(attended)
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    # Define input shape based on the data preprocessing (for example: 100 timesteps, 10 features)
    input_shape = (100, 10)
    model = build_attention_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Generate dummy training data for demonstration purposes.
    X_train = np.random.rand(1000, 100, 10)
    y_train = np.random.randint(0, 2, size=(1000, 1))
    
    model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32)
    
    os.makedirs("NN_winner", exist_ok=True)
    model.save("NN_winner/crypto_model.hdf5")
    print("Saved new crypto model as NN_winner/crypto_model.hdf5")