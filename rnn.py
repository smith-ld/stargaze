import numpy as np
import keras
from keras import layers
from keras.src.layers import SimpleRNN


def create_model(n_dims, vocab_size):
    model = keras.Sequential()
    # output embedding dimension of size 64.
    # input dim = |V|
    # output dim = |columns of E|
    model.add(SimpleRNN(128, return_sequences=True,input_shape=(10, n_dims)))

    # model.add(layers.LSTM(3))
    model.add(layers.SimpleRNN(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    # model.add(layers.Dense(1))
    return model



def train(model: keras.Sequential, xtrain: np.ndarray, ytrain: np.ndarray):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print(xtrain.shape, ytrain.shape)
    # xtrain.shape[1] == num dims for predictions
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)
    return model


