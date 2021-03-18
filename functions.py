#%% Imports
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from models import (
    encoder_decoder1,
    encoder_decoder2,
    encoder_decoder3,
    FNN_model,
)
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
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
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from nbeats_keras.model import NBeatsNet


#%% Functions


#% Data Transformations

# difference dataset
def difference(data, interval=1):
    return (
        [data[i] - data[i - interval] for i in range(interval, len(data))],
        data[0],
    )


def add_difference(data):
    data = np.array(data)
    first_diff, _ = difference(data, interval=1)
    second_diff, _ = difference(first_diff, interval=1)
    return np.concatenate((data[2:], first_diff[1:], second_diff), axis=1)


# invert difference
def invert_difference(orig_data, diff_data, interval=1):
    return [diff_data[i] + orig_data for i in range(interval, len(diff_data))]


def logDiff(data):
    data = np.log(data)
    data, _ = difference(data)
    return data


def normalise(data):
    # fit transform
    transformer = MinMaxScaler()
    transformer.fit(data)
    # normalise
    transformed = transformer.transform(data)
    return transformed, transformer


def unnormalise(transformed, transformer):
    inverted = transformer.inverse_transform(transformed)
    return inverted


# invert a boxcox transform
def invert_boxcox(value, lam):
    # log case
    if lam == 0:
        return np.exp(value)
    # all other cases
    return np.exp(np.log(lam * value + 1) / lam)


def completeTransformation(data):
    boxcoxedData, lmbda = boxcox(data)
    differencedData, orig_data = difference(boxcoxedData)
    normalisedData, transformer = normalise(differencedData)
    return normalisedData, lmbda, orig_data, transformer


def completeInverseTransformation(
    transformedData, lmbda, orig_data, transformer
):
    unnormalisedData = unnormalise(transformedData, transformer)
    undifferencedData = invert_difference(orig_data, unnormalisedData)
    unboxcoxedData = invert_boxcox(undifferencedData, lmbda)
    return unboxcoxedData


#% Model functions

# evaluate one or more quaterly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score


# split dataset into training and testing parts
def split_dataset(data, percentage, n_input=0):
    sizeOfData = data.shape[0]
    splitSize = int(sizeOfData * percentage)
    if n_input:
        train, test = data[1:splitSize], data[splitSize - n_input :]
    else:
        train, test = data[1:splitSize], data[splitSize:]
    return train, test


# convert testing dataset into inputs and outputs (NEW)
def to_supervised(trainset, n_input, n_out=1):
    data = np.array(trainset)
    X, y = list(), list()
    # step over the entire history one time step at a time
    for in_start in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end + n_out - 1 : out_end, 0])
    return np.array(X), np.array(y)


# convert testing dataset into inputs and outputs (OLD)
def to_supervised_old(trainset, n_input, n_out=1):
    data = np.array(trainset)
    X, y = list(), list()
    # step over the entire history one time step at a time
    for in_start in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
    return np.array(X), np.array(y)


# train the model
def build_model(
    trainset,
    testset,
    n_input,
    parameters=[0, 50, 16, 200, 7, 0, 0, 0],
    lr=0.001,
):
    # define parameters
    (
        verbose,
        epochs,
        batch_size,
        latent_dim1,
        n_out,
        l1,
        l2,
        earlystoppingv,
    ) = parameters
    # prepare data
    train_x, train_y = to_supervised(trainset, n_input, n_out)
    # data dependent parameters
    n_timesteps, n_features, n_outputs = (
        train_x.shape[1],
        train_x.shape[2],
        train_y.shape[1],
    )
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    (test_x, test_y) = testset
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    testset = (test_x, test_y)
    # define model

    model = encoder_decoder2(
        latent_dim1, n_outputs, n_timesteps, n_features, l1, l2,
    )
    # model = NBeatsNet(
    #     backcast_length=n_timesteps,
    #     forecast_length=n_outputs,
    #     stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    #     nb_blocks_per_stack=2,
    #     thetas_dim=(4, 4),
    #     share_weights_in_stack=True,
    #     hidden_layer_units=latent_dim1,
    # )
    # model = FNN_model(n_outputs, n_timesteps, n_features)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(
        loss="mse",
        optimizer=adam,
        metrics=["mean_squared_error", "mean_absolute_error"],
    )
    # RLROP= keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error',\
    #      factor=0.1, patience=100, verbose=1, mode='auto', min_delta=0.0001,\
    #           cooldown=100, min_lr=1e-7)
    if earlystoppingv:
        es = EarlyStopping(
            monitor="val_mean_squared_error",
            mode="min",
            verbose=1,
            patience=300,
        )
        # fit network
        model.summary()
        History = model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=testset,
            callbacks=[es],
            shuffle=False,
        )
    else:
        model.summary()
        History = model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=testset,
            shuffle=False,
        )
    return model, History


def continueBuild_model(
    model,
    trainset,
    testset,
    n_input,
    parameters=[0, 50, 16, 200, 7, 0, 0, 0],
    lr=0.001,
):
    # define parameters
    (verbose, epochs, batch_size, _, n_out, _, _, earlystoppingv,) = parameters
    # prepare data
    train_x, train_y = to_supervised(trainset, n_input, n_out)

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    (test_x, test_y) = testset
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    testset = (test_x, test_y)
    # RLROP= keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error',\
    #      factor=f, patience=100, verbose=1, mode='auto',\
    #           min_delta=0.0001, cooldown=1, min_lr=1e-7)
    if earlystoppingv:
        es = EarlyStopping(
            monitor="val_mean_squared_error",
            mode="min",
            verbose=1,
            patience=500,
        )
        # fit network
        History = model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=testset,
            callbacks=[es],
            shuffle=True,
        )
    else:
        History = model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=testset,
            callbacks=[],
            shuffle=True,
        )
    return model, History


# make a forecast
def forecast(model, testset, n_input):

    data = np.array(testset)
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n_features]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(
    train,
    test,
    n_input,
    n_out,
    modelParameters=[0, 50, 16, 200, 7, 0, 0, 0],
    logDiff=0,
    lr=0.001,
):

    # log transform and difference data
    if logDiff:
        train = logDiff(train)
        test = logDiff(test)
    # fit model
    test_x, test_y = to_supervised(test, n_input, n_out)
    validationData = (test_x, test_y)
    model, History = build_model(
        train, validationData, n_input, modelParameters, lr
    )
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test_x)):
        # predict the n_out quarters
        yhat_sequence = forecast(model, test_x[i], n_input)
        # store the predictions
        predictions.append(yhat_sequence)
    # evaluate predictions for each quarter
    predictions = np.array(predictions)
    score = evaluate_forecasts(test_y, predictions)
    return predictions, score, model, History


def continueTraining_model(
    model,
    train,
    test,
    n_input,
    n_out,
    modelParameters=[0, 50, 16, 200, 7, 0, 0, 0],
    logDiff=0,
    lr=0.001,
):

    # log transform and difference data
    if logDiff:
        train = logDiff(train)
        test = logDiff(test)

    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.01)
    model.compile(
        loss="mse",
        optimizer=adam,
        metrics=["mean_squared_error", "mean_absolute_error"],
    )
    # fit model
    test_x, test_y = to_supervised(test, n_input, n_out)
    validationData = (test_x, test_y)
    model, History = continueBuild_model(
        model, train, validationData, n_input, modelParameters, lr
    )
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test_x)):
        # predict the n_out quarters
        yhat_sequence = forecast(model, test_x[i], n_input)
        # store the predictions
        predictions.append(yhat_sequence)
    # evaluate predictions for each quarter
    predictions = np.array(predictions)
    score = evaluate_forecasts(test_y, predictions)
    return predictions, score, model, History


# %% "Super learner"


# create a list of base-models
def get_models():
    models = list()
    models.append(ElasticNet())
    models.append(DummyRegressor())
    models.append(ExtraTreesRegressor(n_estimators=10))
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    models.append(SVR(gamma='scale'))
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(BaggingRegressor(n_estimators=10))
    models.append(DecisionTreeRegressor())
    models.append(LinearRegression())
    models.append(MLPRegressor(max_iter=1000))
    return models


# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=10, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        # get data
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        meta_y.extend(test_y)
        # fit and make predictions with each sub-model
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            # store columns
            fold_yhats.append(yhat.reshape(len(yhat), 1))
        # store fold yhats as columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)


# fit all base models on the training dataset
def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)


# fit a meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        print("%s: RMSE %.7f" % (model.__class__.__name__, sqrt(mse)))


# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat), 1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)


def SuperLearner(test, train):
    X_val, y_val = test
    X, y = train
    # get models
    models = get_models()
    # get out of fold predictions
    meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
    print("Meta ", meta_X.shape, meta_y.shape)
    # fit base models
    fit_base_models(X, y, models)
    # fit the meta model
    meta_model = fit_meta_model(meta_X, meta_y)
    # evaluate base models
    evaluate_models(X_val, y_val, models)
    # evaluate meta model
    yhat = super_learner_predictions(X_val, models, meta_model)
    print("Super Learner: RMSE %.7f" % (sqrt(mean_squared_error(y_val, yhat))))
    return meta_model, models
