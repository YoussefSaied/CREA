#%% Imports
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import importlib, sys
from CREA.Functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    SuperLearner,
    super_learner_predictions,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
)

# importlib.reload(sys.modules['Functions'])
from CREA.DataPreparation import dataPreparation
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, GRU, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from math import sqrt
from numpy import hstack, vstack, asarray
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from tpot import TPOTRegressor

#%% Importing Prof. Nilles's data file on disk.

Nillesdata_filename = "/home/youssef/EPFL/Research/Data/Nillesdata.xlsx"
Nillesdata_xls = pd.ExcelFile(Nillesdata_filename)
Nillesdata = pd.read_excel(
    Nillesdata_filename, sheet_name=Nillesdata_xls.sheet_names[1]
)

# Dropping Quarter column
Quarter = Nillesdata["Quarter"].copy()
Nillesdata = Nillesdata.drop("Quarter", axis=1)


#%% Additional Data preparation and Paremeters

importlib.reload(sys.modules["DataPreparation"])
from DataPreparation import dataPreparation

#% Parameters
# Dimensions (features) of the multivariate time series (max 18)
# set(np.linspace(0,17,num=18,dtype=int).flatten()) {0,1,17}
input_dimSet = set(np.linspace(0, 22, num=23, dtype=int))
input_dimSet = list({0} | input_dimSet)

pretrainLogDiff = False
trainLogDiff = not pretrainLogDiff and False
differencing = pretrainLogDiff or trainLogDiff
data = Nillesdata.copy()

# logDiffCol=[0,2,3,8,9,10,11,12,13,15,16,17,23]
# logCol=[0,8,9,10,11,12,15,16,17,23]
treatedIndices = [0, 8, 9, 10, 11, 12, 15, 16, 17, 23]
logDiffColCondition = False
logColCondition = True
removeMean = False
normaliseData = False
dlGDPtransform = True
MakeClimatPositive = False
SwitchNACSA = True
scaleCondition = True


data = dataPreparation(
    data,
    input_dimSet,
    pretrainLogDiff,
    trainLogDiff,
    treatedIndices,
    logDiffColCondition,
    logColCondition,
    removeMean,
    normaliseData,
    dlGDPtransform,
    MakeClimatPositive,
    SwitchNACSA,
    scaleCondition,
)


#%% General parameters and Data splitting

# Parameters
n_input = 4  # Number of lags to consider 6
n_outputs = 1  # Number of predictive steps

# Data splitting
train, test = split_dataset(data, 0.8, n_input=n_input)
train, test = (
    np.array(train)[:, list(input_dimSet)],
    np.array(test)[:, input_dimSet],
)

#%% Actual training: RNN model


importlib.reload(sys.modules["Functions"])
from Functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    SuperLearner,
    super_learner_predictions,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
)

# Model sepcific parameters:
# latent_dim =2
# l =0.85e-4
batch_size = 16  # Batch size for training. 32 normally
epochs = 4000  # Number of epochs to train for.
latent_dim1 = 2  # Latent dimensionality of the encoding space.
latent_dim2 = 7  # Latent dimensionality of the decoding space.
latent_dim3 = 4  # Latent dimensionality of the output layer.
l1 = 0.00000
l2 = 0.00000  # best: 1000e-5 (16,16) 4, 25e-5 (16,16) 1
lR1 = 0.00
lR2 = 0.00
verbose = 0
earlystopping = False
lr = 0.0001

continueTraining = False

modelParameters = [
    verbose,
    epochs,
    batch_size,
    latent_dim1,
    latent_dim2,
    latent_dim3,
    n_outputs,
    l1,
    l2,
    lR1,
    lR2,
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


diff = predictions.squeeze() - np.array(test)[-predictions.shape[0] :][:, 0]

# Plot predictions
plt.plot(np.squeeze(predictions), label="Predictions")
plt.plot(np.array(test)[-predictions.shape[0] :][:, 0], label="dlGDP")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

# Plot all quarters

predictionsAll = model.predict(
    to_supervised(np.array(data)[:, list(input_dimSet)], n_input, n_outputs)[
        0
    ],
    verbose=0,
)

plt.plot(np.squeeze(predictionsAll), label="All quarters predictions")
plt.plot(np.array(data)[-1 * predictionsAll.shape[0] :][:, 0], label="dlGDP")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

# Plot history
# plt.plot(history.history['loss'], label='Loss (testing data)')
# plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.plot(history.history["mean_squared_error"], label="MSE (testing data)")
plt.plot(
    history.history["val_mean_squared_error"], label="MSE (validation data)"
)
plt.title("MSE for GDP forecasts")
plt.ylabel("MSE value")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.yscale("log")
plt.show()
print(
    "PARAMETERS: "
    + " Batch size = "
    + str(batch_size)
    + " ,(latent dim1,latent_dim3 )= "
    + str((latent_dim1, latent_dim3))
    + " ,Features = "
    + str(input_dimSet)
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
plt.acorr(diff - np.mean(diff), label="dlGDP_csa")
plt.title("diff")
plt.ylabel("autocorrelation")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()


# %% "Super learner"

importlib.reload(sys.modules["Functions"])
from CREA.Functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    SuperLearner,
    super_learner_predictions,
    to_supervised,
    forecast,
    continueTraining_model,
    evaluate_forecasts,
)


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

plt.plot(np.squeeze(predictionsSL), label="Predictions")
plt.plot(np.array(test)[-32:][:, 0], label="dlGDP_csa")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP_csa")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

# Plot all quarters

data_supervised = to_supervised(
    np.array(data)[:, list(input_dimSet)], n_input, n_outputs
)[0]
data_supervised = data_supervised.reshape(
    (
        np.array(data_supervised).shape[0],
        np.array(data_supervised).shape[1]
        * np.array(data_supervised).shape[2],
    )
)

predictionsAll = super_learner_predictions(data_supervised, models, meta_model)


plt.plot(np.squeeze(predictionsAll), label="All quarters predictions")
plt.plot(np.array(data)[-1 * predictionsAll.shape[0] :][:, 0], label="dlGDP")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

# Autocorrelation
plt.acorr(np.squeeze(diffSL - np.mean(diffSL)), label="dlGDP_csa")
plt.title("diff autocorrelation")
plt.ylabel("autocorrelation")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()


#%% recursive predictions
predictionsRec = []
for i in range(len(testSL[0])):
    meta_model, models = SuperLearner(testSL, trainSL)
    datapoint = testSL[0][i]
    datapoint.shape = (1, testSL[0][i].size)
    datapointLabel = testSL[1][i]
    datapointLabel.shape = (1, testSL[1][i].size)
    prediction = super_learner_predictions(datapoint, models, meta_model)
    trainSLX = np.concatenate((trainSL[0], datapoint))
    trainSLy = np.concatenate((trainSL[1], datapointLabel))
    trainSL = (trainSLX, trainSLy)
    predictionsRec.append(prediction)

predictionsRec = np.array(predictionsRec)[np.newaxis]
#%% Hyperparameters gridsearch

epochs = 1500  # Number of epochs to train for.
n_outputs = 1  # Number of predictive steps
verbose = 0
batch_size = 32  # Batch size for training.

latent_dimList = np.geomspace(2, 2 ** 7, num=7, dtype=int)
n_inputList = np.linspace(1, 8, 8, dtype=int)
lList = np.geomspace(2 ** -17, 2 ** -10, num=7, dtype=float)
# lList2 =np.geomspace(2**-17,2**-10,num=7,dtype=float)
resultsList = list()

for l, latent_dim, n_input in itertools.product(
    lList, latent_dimList, n_inputList
):
    train, test = split_dataset(data, 0.8, n_input=n_input)
    train, test = (
        np.array(train)[:, list(input_dimSet)],
        np.array(test)[:, input_dimSet],
    )
    if pretrainLogDiff:
        train = logDiff(train)
        test = logDiff(test)
    modelParameters = [
        verbose,
        epochs,
        batch_size,
        latent_dim,
        latent_dim,
        n_outputs,
        l,
        l,
    ]
    predictions, score, model, History = evaluate_model(
        train, test, n_input, modelParameters, logDiff=trainLogDiff
    )
    resultsList.append((np.array([l, latent_dim, n_input]), score))
    print(
        "PARAMETERS: "
        + " Batch size = "
        + str(batch_size)
        + " ,(latent dim1,latent_dim2 )= "
        + str((latent_dim, latent_dim))
        + " ,Features = "
        + str(input_dimSet)
        + " ,n_inputs = "
        + str(n_input)
        + " ,l1 = "
        + str(l)
        + " ,l2 = "
        + str(l)
    )
    print(
        "score: "
        + str(score)
        + " ,val_lose = "
        + str(history.history["val_loss"][-1])
    )


#%% Data analysis

# Plot GDP after normalization
plt.plot(data["GDP_csa"], label="GDP_csa")
# plt.plot(data["GDP_na"], label='GDP_na')
plt.title("Transformed GDP")
plt.ylabel("GDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

# Crosscorrelation
column_names = ["lags"] + list(data.columns)
Crosscorrelationdf = pd.DataFrame(columns=column_names)
lags, _, _, _ = plt.xcorr(data["GDP_csa"], data["GDP_csa"])
Crosscorrelationdf["lags"] = lags
for col in data.columns:
    standardisedGDP = np.array((data["GDP_csa"] - data["GDP_csa"].mean()))
    standardisedCol = np.array(data[col] - data[col].mean())
    _, corr, _, _ = plt.xcorr(standardisedGDP, standardisedCol)
    Crosscorrelationdf[col] = corr

#%% GDP per employed person

GDPperEmployed = Nillesdata["GDP_csa"] / Nillesdata["PAO"]
column_names = ["Quarter", "PeGDP"]
GDPperEmployeddf = pd.DataFrame(columns=column_names)
GDPperEmployeddf["Quarter"] = Quarter
GDPperEmployeddf["PeGDP"] = GDPperEmployed


plt.plot(GDPperEmployed, label="PeGDP")
plt.title("GDP per employed person")
plt.ylabel("PeGDP")
plt.xlabel("Quarter")
# plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

#%% Differencing plots

GDP_csa = np.array(Nillesdata["GDP_csa"])
lGDP_csa = np.log(GDP_csa)
dlGDP_csa, _ = difference(lGDP_csa)
dlGDP_csa = np.array(dlGDP_csa)
ddlGDP_csa, _ = difference(dlGDP_csa)

GDP_na = np.array(Nillesdata["GDP_na"])
lGDP_na = np.log(GDP_na)
dlGDP_na, _ = difference(lGDP_na)
ddlGDP_na, _ = difference(dlGDP_na)


# plt.plot(dlGDP_csa, label='dlGDP_csa')
plt.plot(dlGDP_na, label="dlGDP_na")
plt.title("dlGDP")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

plt.plot(ddlGDP_csa, label="ddlGDP_csa")
# plt.plot(ddlGDP_na, label='ddlGDP_na')
plt.title("ddlGDP")
plt.ylabel("ddlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

plt.plot(np.array(dlGDP_csa) - np.array(dlGDP_na), label="GDP_csa-GDP_na")
plt.title("GDP_csa-GDP_na")
plt.ylabel("GDP_csa-GDP_na")
plt.xlabel("Quarter")
# plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

# Autocorrelation
plt.acorr(dlGDP_csa - np.mean(dlGDP_csa), label="dlGDP_csa")
# plt.acorr(dlGDP_na, label='dlGDP_na')
plt.title("dlGDP autocorrelation")
plt.ylabel("autocorrelation")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

plt.acorr(ddlGDP_csa - np.mean(ddlGDP_csa), label="ddlGDP_csa")
# plt.acorr(dlGDP_na, label='dlGDP_na')
plt.title("ddlGDP autocorrelation")
plt.ylabel("autocorrelation")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.show()

#%% Smooth plots

plt.plot(dlGDP_csa[1:], label="dlGDP")
smooth2 = (dlGDP_csa[:-1] + dlGDP_csa[1:]) / 2
plt.plot((dlGDP_csa[:-1] + dlGDP_csa[1:]) / 2, label="dlGDP smooth2")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

plt.plot(dlGDP_csa[1:-1], label="dlGDP")
smooth3 = 0.25 * dlGDP_csa[:-2] + 0.5 * dlGDP_csa[1:-1] + 0.25 * dlGDP_csa[2:]
plt.plot(smooth3, label="dlGDP smooth3")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

plt.plot(dlGDP_csa[2:-2], label="dlGDP")
smooth5 = (
    dlGDP_csa[:-4]
    + dlGDP_csa[1:-3]
    + dlGDP_csa[2:-2]
    + dlGDP_csa[3:-1]
    + dlGDP_csa[4:]
) / 5
plt.plot(smooth5, label="dlGDP smooth5")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

plt.plot(dlGDP_csa[1:-1] - smooth3, label="dlGDP- smooth3")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()

plt.plot(dlGDP_csa[2:-2] - smooth5, label="dlGDP- smooth5")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()


# %% tpot

testSL = to_supervised(test, n_input, n_outputs)
trainSL = to_supervised(train, n_input, n_outputs)
testSL[0].shape = (testSL[0].shape[0], testSL[0].shape[1] * testSL[0].shape[2])
trainSL[0].shape = (
    trainSL[0].shape[0],
    trainSL[0].shape[1] * trainSL[0].shape[2],
)
(X_train, y_train) = trainSL
(X_test, y_test) = testSL
tpot = TPOTRegressor(
    generations=20, population_size=100, verbosity=2, random_state=42
)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export("tpot_boston_pipeline.py")

# %% Plot
predictions = tpot.predict(X_test)
plt.plot(np.squeeze(predictions), label="Predictions")
plt.plot(np.array(test)[-1 * predictions.shape[0] :][:, 0], label="dlGDP_csa")
plt.title("dlGDP forecasts")
plt.ylabel("dlGDP_csa")
plt.xlabel("Quarter")
plt.legend(loc="upper left")
plt.show()
