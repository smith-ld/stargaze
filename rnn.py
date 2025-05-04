import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.src.layers import SimpleRNN
from nltk import WhitespaceTokenizer

SEQUENCE_LENGTH = 10

VOCAB_SIZE = 100

OUTPUT_DIMENSION = 1
NUM_DIMS = 107


def create_model(n_dims, vocab_size):
    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    # input dim = |V|
    # output dim = |columns of E|
    model.add(SimpleRNN(50, input_shape=(10, n_dims)))

    # Add a LSTM layer with 128 internal units.
    # model.add(layers.LSTM(3))

    # Add a Dense layer with 10 units.

    model.add(layers.Dense(vocab_size, activation='relu'))
    # model.add(layers.Dense(1))
    return model



def train(model: keras.Sequential, xtrain: np.ndarray, ytrain: np.ndarray):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print(xtrain.shape, ytrain.shape)
    # xtrain.shape[1] == num dims for predictions
    model.fit(xtrain, ytrain, epochs=100, batch_size=32)
    return model



# import spacy
# nlp = spacy.load("en_core_web_sm")
#
# doc = nlp(texts[0])
# for token in doc:
#     print(token.vector.shape, token.pos_, token.tag_, token.dep_)