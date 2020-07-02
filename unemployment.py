#%% Basic functions

import os
import importlib, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CREA.functions import (
    continueTraining_model,
    difference,
    evaluate_forecasts,
    evaluate_model,
    forecast,
    logDiff,
    normalise,
    split_dataset,
    super_learner_predictions,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
    add_difference,
)
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


def load_data_unemployemnt():
    """
    split: Train/ Test split ratio


    Return
    - train input (N*r x 1) sequence
    - train target  (N*r) sequence
    - test input (N*(1-r) x 1) sequence
    - test target (N*(1-r) x 1) sequence

    """

    data_dir = os.environ.get("PYTORCH_DATA_RNN_DIR")
    if data_dir is None:
        data_dir = "Data/"

    unemployment_data_filename = data_dir + "UNRATE.csv"
    unemployment_data = pd.read_csv(unemployment_data_filename)

    # Dropping Quarter column
    months = unemployment_data["DATE"].copy()
    unemployment_data = unemployment_data.drop("DATE", axis=1)

    return unemployment_data, months


#%% General parameters and Data splitting

importlib.reload(sys.modules["functions"])
from functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
    add_difference,
)

# Parameters
n_input = 36  # Number of lags to consider 36
n_outputs = 1  # Number of predictive steps
add_diff = True
use_diff = False
log = False
norm = False

# Data import and splitting
data_undifferenced, _ = load_data_unemployemnt()
data_undifferenced = np.array(data_undifferenced)
if use_diff:
    data, _ = difference(data_undifferenced, interval=1)
    data = np.array(data)
else:
    data = data_undifferenced
train, test = split_dataset(data, 450.0 / 660.0, n_input=n_input)
if norm:
    train, transformer = normalise(train)
    test = transformer.transform(test)
    data = transformer.transform(data)

if add_diff:
    data = add_difference(data)
    train, test = split_dataset(data, 450.0 / 660.0, n_input=n_input)

if log:
    data_undifferenced = np.log(data_undifferenced)


#%% Analysis

importlib.reload(sys.modules["models"])
importlib.reload(sys.modules["functions"])
from functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
    add_difference,
)


# Model sepcific parameters:
batch_size = 256  # Batch size for training. 32 normally max 400
epochs = 3000  # Number of epochs to train for.
latent_dim1 = 50  # Latent dimensionality of the encoding space.

l1 = 0.00
l2 = 0  # best: 1000e-5 (16,16) 4, 25e-5 (16,16) 1 # 01/07 best l2=1e-4 
#02/07 0.177(dropout=0.01) 0.1499(dropout=0.1, latent =5)

verbose = 2
earlystopping = True
lr = 1e-4
trainLogDiff = 0

continueTraining = False

modelParameters = [
    verbose,
    epochs,
    batch_size,
    latent_dim1,
    n_outputs,
    l1,
    l2,
    earlystopping,
]


# Training
if not continueTraining:
    predictions, score, model, history = evaluate_model(
        train,
        test,
        n_input,
        n_outputs,
        modelParameters,
        logDiff=trainLogDiff,
        lr=lr,
    )
else:
    predictions, score, model, history = continueTraining_model(
        model,
        train,
        test,
        n_input,
        n_outputs,
        modelParameters,
        logDiff=trainLogDiff,
        lr=lr,
    )


predictionsAll = model.predict(
    to_supervised(np.array(data)[:], n_input, n_outputs,)[0], verbose=0,
).reshape((-1, 1))
predictions = predictionsAll[-1 * predictions.shape[0] :]

# if norm:
#     predictionsAll = transformer.inverse_transform(
#         predictionsAll.reshape((-1, 1))
#     )

if norm:
    print(
        "MAE = "
        + str(
            mean_absolute_error(
                transformer.inverse_transform(predictions.reshape((-1, 1))),
                transformer.inverse_transform(
                    np.array(test)[-predictions.shape[0] :][:, 0].reshape(
                        (-1, 1)
                    )
                ),
            ),
        )
    )

# Plot predictions
plt.plot(
    np.squeeze(predictions),
    marker=".",
    linewidth=1,
    markersize=2,
    label="Predictions",
)

realRates = np.array(data)[-predictions.shape[0] :][:, 0].reshape((-1, 1))
plt.plot(
    realRates, marker=".", linewidth=1, markersize=2, label="Unemployment",
)
plt.title("Unemployment forecasts")
plt.ylabel("Unemployment")
plt.xlabel("Month")
plt.legend(loc="upper left")
plt.show()

# Plot all months


plt.plot(
    np.squeeze(predictionsAll),
    marker=".",
    linewidth=1,
    markersize=2,
    label="All Unemployment predictions",
)
plt.plot(
    np.array(data)[-1 * predictionsAll.shape[0] :][:, 0],
    marker=".",
    linewidth=1,
    markersize=2,
    label="Unemployment",
)
plt.title("Unemployment forecasts")
plt.ylabel("Unemployment rate")
plt.xlabel("Month")
plt.legend(loc="upper left")
plt.show()

# Plot history
plt.plot(history.history["mean_absolute_error"], label="MAE (testing data)")
plt.plot(
    history.history["val_mean_absolute_error"], label="MAE (validation data)"
)
plt.title("MAE for Unemployment forecasts")
plt.ylabel("MAE value")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.yscale("log")
plt.show()
print(
    "PARAMETERS: "
    + " Batch size = "
    + str(batch_size)
    + " ,(latent dim1 )= "
    + str((latent_dim1))
    + " ,Features = "
    + str(n_input)
    + " ,n_inputs = "
    + str(n_input)
    + " ,l1 = "
    + str(l1)
    + " ,l2 = "
    + str(l2)
    + ", lr = "
    + str(lr)
)
print(
    "score: "
    + str(score)
    + " ,val_lose = "
    + str(history.history["val_loss"][-1])
)

# print('diff:')
# print(diff)
# print('predictions:')
# print(predictions)

#% Diff analysis

# Autocorrelation
diff = predictions.squeeze() - np.array(test)[-predictions.shape[0] :][:, 0]

plt.acorr(diff - np.mean(diff), label="Unemployment")
plt.title("diff")
plt.ylabel("autocorrelation")
plt.xlabel("Month")
plt.legend(loc="upper left")
plt.show()


#%% Grid search

importlib.reload(sys.modules["models"])
importlib.reload(sys.modules["functions"])
from functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
    add_difference,
)


# Model sepcific parameters:
batch_size = 128  # Batch size for training. 32 normally max 400
epochs = 1500  # Number of epochs to train for.
latent_dim1_list =  np.linspace(3,10,8,dtype=int)  # Latent dimensionality of the encoding space.

l1 = 0.00
l2 = 1e-4

verbose = 2
earlystopping = True
lr = 1e-3
trainLogDiff = 0

continueTraining = False


scores = []

for latent_dim1 in latent_dim1_list:
    modelParameters = [
        verbose,
        epochs,
        batch_size,
        latent_dim1,
        n_outputs,
        l1,
        l2,
        earlystopping,
    ]
    predictions, score_int, model, history = evaluate_model(
        train,
        test,
        n_input,
        n_outputs,
        modelParameters,
        logDiff=trainLogDiff,
        lr=lr,
    )
    score=score_int
    predictions, score_int, model, history = evaluate_model(
        train,
        test,
        n_input,
        n_outputs,
        modelParameters,
        logDiff=trainLogDiff,
        lr=lr,
    )
    score =(score +score_int)/2
    scores = scores + [score]
    # Plot history
    plt.plot(history.history["mean_absolute_error"], label="MAE (testing data)")
    plt.plot(
        history.history["val_mean_absolute_error"], label="MAE (validation data)"
    )
    plt.title("MAE for Unemployment forecasts")
    plt.ylabel("MAE value")
    plt.xlabel("No. epoch")
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.show()
    print(
        "PARAMETERS: "
        + " Batch size = "
        + str(batch_size)
        + " ,(latent dim1 )= "
        + str((latent_dim1))
        + " ,Features = "
        + str(n_input)
        + " ,n_inputs = "
        + str(n_input)
        + " ,l1 = "
        + str(l1)
        + " ,l2 = "
        + str(l2)
        + ", lr = "
        + str(lr)
    )
    print(
        "score: "
        + str(score)
        + " ,val_lose = "
        + str(history.history["val_loss"][-1])
    )
    tf.keras.backend.clear_session()
    


    

plt.plot(
    latent_dim1_list[:],
    np.squeeze(scores),
    marker=".",
    linewidth=1,
    markersize=2,
    label="Scores for latent dimension",
)

plt.title("Hyper-parameter to adjust")
plt.ylabel("Score ")
plt.xlabel("latent_dim1")
plt.legend(loc="upper left")
plt.show()
