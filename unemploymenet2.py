# %% Basic functions

import os
import random
import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions_marco import (
    gen_sequence,
    gen_labels,
    train_test_split_marco,
    autoencoder_marco,
    forecaster_marco
)
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_data_unemployemnt():
    """
    loads unmployment timeseries for OECD

    Return
    - unemployment sequence
    - months sequence

    """

    data_dir = os.environ.get("PYTORCH_DATA_RNN_DIR")
    if data_dir is None:
        data_dir = "Data/"

    unemployment_data_filename = data_dir + "OECD_unemployment.csv"
    unemployment_data = pd.read_csv(unemployment_data_filename)

    return unemployment_data


# %% Data loading and splitting
importlib.reload(sys.modules["functions_marco"])


df = load_data_unemployemnt()
dfm = df[(df.FREQUENCY == 'M')]
dfm = dfm.filter(['LOCATION', 'TIME', 'Value', 'SUBJECT'])
dfm = dfm.reset_index(drop=True)

sequence_length = 36
X_train, X_test, y_train, y_test, X_other_test, y_other_test = train_test_split_marco(
    dfm, sequence_length)


# %% Creating the encoder
importlib.reload(sys.modules["functions_marco"])


### SET SEED ###
seed = 42
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)

tf.compat.v1.keras.backend.set_session(sess)


### TRAIN AUTOENCODER ###

sequence_autoencoder, encoder = autoencoder_marco(sequence_length)
history = sequence_autoencoder.fit(X_train, X_train,
                                   batch_size=128, epochs=200, verbose=1, shuffle=True, validation_data=(X_test, X_test))

# %% Encoding sequences

XX_train = encoder.predict(X_train)
XX_test = encoder.predict(X_test)

# XX_train = X_train
# XX_test = X_test

# XX_other_test = encoder.predict(X_other_test)

# %% Forecaster

### SCALE DATA ###

scaler1 = StandardScaler()
X_train1 = scaler1.fit_transform(
    XX_train.reshape(-1, XX_train.shape[-1])).reshape(-1, sequence_length, XX_train.shape[-1])
X_test1 = scaler1.transform(
    XX_test.reshape(-1, XX_test.shape[-1])).reshape(-1, sequence_length, XX_test.shape[-1])
# X_other_test1 = scaler1.transform(
#     XX_other_test.reshape(-1, XX_other_test.shape[-1])).reshape(-1, sequence_length, XX_other_test.shape[-1])


### TRAIN FORECASTER ###
forecaster = forecaster_marco(X_train1)
history = forecaster.fit(X_train1, y_train, epochs=450,
                         batch_size=128, verbose=2, shuffle=True, validation_data=(X_test1, y_test))

# %% Test


predictions = forecaster.predict(X_test1)
score = mean_absolute_error(predictions, y_test)

plt.plot(np.squeeze(predictions), marker=".",
         linewidth=1, markersize=2, label="Predictions")
plt.plot(y_test, marker=".", linewidth=1,
         markersize=2, label="unemployment")
plt.title("unemployment forecasts")
plt.ylabel("unemployment")
plt.xlabel("months")
plt.legend(loc="upper left")
plt.show()
