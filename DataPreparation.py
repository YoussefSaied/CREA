import numpy as np
import pandas as pd
from Functions import (
    difference,
    evaluate_model,
    split_dataset,
    logDiff,
    normalise,
    SuperLearner,
    super_learner_predictions,
)


def dataPreparation(
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
):

    trainLogDiff = not pretrainLogDiff and trainLogDiff
    differencing = pretrainLogDiff or trainLogDiff

    if pretrainLogDiff and not trainLogDiff:
        data = logDiff(data)

    # Making climate positive
    if MakeClimatPositive or (differencing and 1 in input_dimSet):
        data["CLIMAT"] = np.abs(data["CLIMAT"].min()) * 1.3 + data["CLIMAT"]

    # Transforming Data

    if logDiffColCondition and not differencing and not logColCondition:
        ldData = logDiff(np.array(data.iloc[:, treatedIndices]))
        data = data.drop(0, axis=0)
        data.iloc[:, treatedIndices] = ldData
        if removeMean:
            data["GDP_csa"] = data["GDP_csa"] - np.mean(data["GDP_csa"][:-31])
            data["GDP_na"] = data["GDP_na"] - np.mean(data["GDP_na"][:-31])

    if normaliseData:
        _, transformer = normalise(np.array(data[:-31]))
        data[list(data.columns)] = transformer.transform(np.array(data))

    if logColCondition and not logDiffColCondition:
        logData = np.log(np.array(data.iloc[:, treatedIndices]))
        data.iloc[:, treatedIndices] = logData
        if removeMean:
            data["GDP_csa"] = data["GDP_csa"] - np.mean(data["GDP_csa"][:-31])
            data["GDP_na"] = data["GDP_na"] - np.mean(data["GDP_na"][:-31])

    if dlGDPtransform:
        if not logDiffColCondition and not logColCondition:
            dlGDPcsa = logDiff(np.array(data["GDP_csa"]))
            dlGDPna = logDiff(np.array(data["GDP_na"]))
            data = data.drop(0, axis=0)
            data["GDP_csa"] = dlGDPcsa
            data["GDP_na"] = dlGDPna
            if removeMean:
                data["GDP_csa"] = data["GDP_csa"] - np.mean(
                    data["GDP_csa"][:-31]
                )
                data["GDP_na"] = data["GDP_na"] - np.mean(data["GDP_na"][:-31])
        if logColCondition and not logDiffColCondition:
            dlGDPcsa, _ = difference(np.array(data["GDP_csa"]))
            dlGDPna, _ = difference(np.array(data["GDP_na"]))
            data = data.drop(0, axis=0)
            data["GDP_csa"] = dlGDPcsa
            data["GDP_na"] = dlGDPna
            if removeMean:
                data["GDP_csa"] = data["GDP_csa"] - np.mean(
                    data["GDP_csa"][:-31]
                )
                data["GDP_na"] = data["GDP_na"] - np.mean(data["GDP_na"][:-31])

    # Switch na and csa

    if SwitchNACSA:
        columns_titles = [
            "GDP_na",
            "CLIMAT",
            "DOLLAR",
            "ERN",
            "HYP",
            "INT",
            "INTEP",
            "INTZH",
            "M1",
            "M2",
            "M3",
            "MBC",
            "PAO",
            "PC",
            "UC",
            "WN",
            "WXR",
            "YOECD",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "S08Q4",
            "GDP_csa",
        ]
        data = data.reindex(columns=columns_titles)

    if scaleCondition:
        data["CLIMAT"] = data["CLIMAT"] * 0.1
        data["ERN"] = data["ERN"] * 0.01
        data["PC"] = data["PC"] * 0.01
        data["GDP_csa"] = data["GDP_csa"] * 100
        data["GDP_na"] = data["GDP_na"] * 100

    return data
