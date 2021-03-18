
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import random
import os

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

### CREATE GENERATOR FOR LSTM WINDOWS AND LABELS ###


def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]


### CREATE TRAIN/TEST PRICE DATA ###
def train_test_split_marco(df, sequence_length):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    y_other_test = []
    X_other_test = []

    for country in df["LOCATION"].unique():

        for sequence in gen_sequence(df[(df["LOCATION"] == country) & (df["LOCATION"] != "USA")
                                        & (df["TIME"] < '2020-01')],
                                     sequence_length, ['Value']):
            # if country != 'USA' else X_other_train.append(sequence)
            X_train.append(sequence)

        for sequence in gen_sequence(df[(df["LOCATION"] == country) & (df["LOCATION"] == "USA")
                                        & (df["TIME"] >= '2000-01') & (df["TIME"] < '2020-01')
                                        & (df.SUBJECT == "TOT")],
                                     sequence_length, ['Value']):
            X_test.append(
                sequence)  # if country != 'USA' else X_other_test.append(sequence)

        for sequence in gen_labels(df[(df["LOCATION"] == country) & (df["LOCATION"] != "USA")
                                      & (df["TIME"] < '2020-01')],
                                   sequence_length, ['Value']):
            # if country != 'USA' else X_other_train.append(sequence)
            y_train.append(sequence)

        for sequence in gen_labels(df[(df["LOCATION"] == country) & (df["LOCATION"] == "USA")
                                      & (df["TIME"] >= '2000-01') & (df["TIME"] < '2020-01')
                                      & (df.SUBJECT == "TOT")],
                                   sequence_length, ['Value']):
            y_test.append(
                sequence)  # if country != 'USA' else y_other_test.append(sequence)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_other_test = np.asarray(X_other_test)
    y_other_test = np.asarray(y_other_test)

    return X_train, X_test, y_train, y_test, X_other_test, y_other_test


### DEFINE LSTM AUTOENCODER ###

def autoencoder_marco(sequence_length):
    inputs_ae = Input(shape=(sequence_length, 1))
    encoded_ae = LSTM(32, return_sequences=False, dropout=0.5)(
        inputs_ae, training=True)
    # print(encoded_ae.shape)
    # encoded_ae = tf.keras.layers.Reshape(
    #     (encoded_ae.shape[1], 1))(encoded_ae)
    decoded_ae = RepeatVector(sequence_length)(encoded_ae)
    decoded_ae = LSTM(16, return_sequences=False, dropout=0.5)(
        decoded_ae, training=True)
    out_ae = TimeDistributed(Dense(1))(decoded_ae)

    sequence_autoencoder = Model(inputs_ae, out_ae)
    encoder = Model(inputs_ae, encoded_ae)
    sequence_autoencoder.compile(
        optimizer='adam', loss='mse', metrics=['mse', 'mean_absolute_error'])

    return sequence_autoencoder, encoder


### DEFINE FORECASTER ###

def forecaster_marco(X_train1):
    inputs1 = Input(shape=(X_train1.shape[1], X_train1.shape[2]))
    lstm1 = LSTM(128, return_sequences=True, dropout=0.5)(
        inputs1, training=True)
    lstm1 = LSTM(128, return_sequences=False,
                 dropout=0.5)(lstm1, training=True)
    dense1 = Dense(50)(lstm1)
    out1 = Dense(1)(dense1)

    model1 = Model(inputs1, out1)
    model1.compile(loss='mse', optimizer='adam',
                   metrics=['mse', 'mean_absolute_error'])
    return model1
