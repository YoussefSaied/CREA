# %% Basic functions

import os
import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import (
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
    loads unmployment timeseries for US

    Return
    - unemployment sequence
    - months sequence
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


# %% General parameters and Data splitting

importlib.reload(sys.modules["functions"])

# Parameters
n_input = 10  # Number of lags to consider 36
n_outputs = 1  # Number of predictive steps
add_diff = False
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


# %% Analysis

importlib.reload(sys.modules["models"])
importlib.reload(sys.modules["functions"])


# Model sepcific parameters:
batch_size = 200  # Batch size for training. 32 normally max 400
epochs = 3000  # Number of epochs to train for.
latent_dim1 = 6  # Latent dimensionality of the encoding space.

l1 = 0.00
l2 = 0  # best: 1000e-5 (16,16) 4, 25e-5 (16,16) 1 # 01/07 best l2=1e-4
# 02/07 0.177(dropout=0.01) 0.1499(dropout=0.1, latent =5)

verbose = 2
earlystopping = True
lr = 1e-3
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
predictions = predictionsAll[-1 * predictions.shape[0]:]

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
                    np.array(test)[-predictions.shape[0]:][:, 0].reshape(
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

realRates = np.array(data)[-predictions.shape[0]:][:, 0].reshape((-1, 1))
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
    np.array(data)[-1 * predictionsAll.shape[0]:][:, 0],
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

# % Diff analysis

# Autocorrelation
diff = predictions.squeeze() - np.array(test)[-predictions.shape[0]:][:, 0]

plt.acorr(diff - np.mean(diff), label="Unemployment")
plt.title("diff")
plt.ylabel("autocorrelation")
plt.xlabel("Month")
plt.legend(loc="upper left")
plt.show()


# %% Grid search

importlib.reload(sys.modules["models"])
importlib.reload(sys.modules["functions"])


# Model sepcific parameters:
batch_size = 128  # Batch size for training. 32 normally max 400
epochs = 1500  # Number of epochs to train for.
# Latent dimensionality of the encoding space.
latent_dim1_list = np.linspace(3, 10, 8, dtype=int)

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
    score = score_int
    predictions, score_int, model, history = evaluate_model(
        train,
        test,
        n_input,
        n_outputs,
        modelParameters,
        logDiff=trainLogDiff,
        lr=lr,
    )
    score = (score + score_int)/2
    scores = scores + [score]
    # Plot history
    plt.plot(history.history["mean_absolute_error"],
             label="MAE (testing data)")
    plt.plot(
        history.history["val_mean_absolute_error"], label="MAE (validation data)"
    )
    plt.title("MAE for Unemployment forecasts")
    plt.ylabel("MAE value")
    plt.xlabel("No. epoch")
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.xscale("log")
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

# %% Super learner


importlib.reload(sys.modules["functions"])


testSL = to_supervised(test, n_input, n_outputs)
trainSL = to_supervised(train, n_input, n_outputs)
testSL[0].shape = (testSL[0].shape[0], testSL[0].shape[1] * testSL[0].shape[2])
trainSL[0].shape = (
    trainSL[0].shape[0],
    trainSL[0].shape[1] * trainSL[0].shape[2],
)

meta_model, models = SuperLearner(testSL, trainSL)
predictionsSL = super_learner_predictions(testSL[0], models, meta_model)

diffSL = predictionsSL - testSL[1]
realRates = np.array(data)[-predictionsSL.shape[0]:][:, 0].reshape((-1, 1))
plt.plot(np.squeeze(predictionsSL), label="Predictions")
plt.plot(realRates, label="unemployment")
plt.title("unemployment forecasts")
plt.ylabel("unemployment")
plt.xlabel("months")
plt.legend(loc="upper left")
plt.show()

# Plot all quarters

data_supervised = to_supervised(
    np.array(data), n_input, n_outputs
)[0]
data_supervised = data_supervised.reshape(
    (
        np.array(data_supervised).shape[0],
        np.array(data_supervised).shape[1]
        * np.array(data_supervised).shape[2],
    )
)

predictionsAll = super_learner_predictions(data_supervised, models, meta_model)


plt.plot(np.squeeze(predictionsAll), label="All months predictions")
plt.plot(np.array(data)[-1 * predictionsAll.shape[0]:]
         [:, 0], label="unemployment")
plt.title("unemployment forecasts")
plt.ylabel("unemployment")
plt.xlabel("months")
plt.legend(loc="upper left")
plt.show()

# Autocorrelation
plt.acorr(np.squeeze(diffSL - np.mean(diffSL)), label="unemployment")
plt.title("diff autocorrelation")
plt.ylabel("autocorrelation")
plt.xlabel("months")
plt.legend(loc="upper left")
plt.show()
