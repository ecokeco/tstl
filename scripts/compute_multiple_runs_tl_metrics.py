
import sys
import os

# Loading Packages
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn import datasets
from scipy.integrate import simps

import argparse
import numpy as np
import h5py
import json
import Metrics
import TransferLearning

baselineModelNames = [
    ("mlstm_fcn-lomax", ("lomax1k5", "lomax9k")),
    ("mlstm_fcn-lendb", ("lendb1k5", "lendb9k")),
    ("mlstm_fcn-stead", ("stead1k5", "stead9k")),
    ("mlstm_fcn-speech8khz", ("speech8khz1k5", "speech8khz9k")),
    ("mlstm_fcn-emg", ("emg1k5", "emg9k")),
    ("mlstm_fcn-sp500", ("sp5001k5", "sp5009k")),
    
    ("tcn-lomax", ("lomax1k5", "lomax9k")),
    ("tcn-lendb", ("lendb1k5", "lendb9k")),
    ("tcn-stead", ("stead1k5", "stead9k")),
    ("tcn-speech8khz", ("speech8khz1k5", "speech8khz9k")),
    ("tcn-emg", ("emg1k5", "emg9k")),
    ("tcn-sp500", ("sp5001k5", "sp5009k")),
    
    ("convnetquakeingv-lomax", ("lomax1k5", "lomax9k")),
    ("convnetquakeingv-lendb", ("lendb1k5", "lendb9k")),
    ("convnetquakeingv-stead", ("stead1k5", "stead9k")),
    ("convnetquakeingv-speech8khz", ("speech8khz1k5", "speech8khz9k")),
    ("convnetquakeingv-emg", ("emg1k5", "emg9k")),
    ("convnetquakeingv-sp500", ("sp5001k5", "sp5009k")),
    
    ("magnet-lomax", ("lomax1k5", "lomax9k")),
    ("magnet-lendb", ("lendb1k5", "lendb9k")),
    ("magnet-stead", ("stead1k5", "stead9k")),
    ("magnet-speech8khz", ("speech8khz1k5", "speech8khz9k")),
    ("magnet-emg", ("emg1k5", "emg9k")),
    ("magnet-sp500", ("sp5001k5", "sp5009k")),
]

targetDatasets = [              # Target datasets
    ("lomax1k5", "regression", 4.3),
    ("lomax9k",  "regression", 4.3),
    ("lendb1k5", "regression", 4.3),
    ("lendb9k", "regression", 4.3),
    ("stead1k5", "regression", 4.3),
    ("stead9k", "regression", 4.3),
    ("speech8khz1k5", "classification", None),
    ("speech8khz9k", "classification", None),
    ("emg1k5", "classification", None),
    ("emg9k", "classification", None),
    ("sp5001k5", "regression", 1960),
    ("sp5009k", "regression", 1960),
]

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
parser = argparse.ArgumentParser()
parser.add_argument("lrMultiplier", help="Processes models that used specified lr multiplier.", default=None)
args = parser.parse_args()

# Set LR multipliers
lrMultiplier = args.lrMultiplier
print("LR multiplier: {}".format(lrMultiplier))

# Compute metrics
processedFilesCount = 0
for run in range(7):
    for baseDataset, taskType, maxTolerance in targetDatasets:
        dataset = baseDataset + "_" + str(run)
        print("Loading dataset {}".format(dataset))
        # Open specified test dataset
        Y_test = np.load(scriptDir + "../data/" + dataset + "_test_Y.npy")
        if taskType == "regression":
            y_true = Y_test[:, 0]
        else:
            y_true = Y_test.argmax(axis=-1)

        for baselineModel, ignoredDatasets in baselineModelNames:
            if baseDataset in ignoredDatasets:
                continue
            modelsName = "{}-{}-{}".format(baselineModel, dataset, lrMultiplier)
            modelsDir = scriptDir + "../models/{}/".format(modelsName)
            if not os.path.exists(modelsDir):
                print("Model {} does not exist".format(modelsName))
            else:
                print("Processing {}".format(modelsName))
            
            predictionsFilename = modelsDir + "predictions.h5"
            predictions = h5py.File(predictionsFilename, "r")
            lastEpoch = int(predictions.attrs["lastEpoch"])
            
            # Determine best epoch
            _, bestEpoch = TransferLearning.getBestModel(modelsName, return_epoch=True)
            
            jsonData = {}
            jsonData["bestEpoch"] = bestEpoch
            jsonData["lastEpoch"] = lastEpoch
            jsonData["mae"] = []
            jsonData["medianabsoluteerror"] = []
            jsonData["mape"] = []
            jsonData["mse"] = []
            jsonData["recauc"] = []
            jsonData["f1"] = []
            for epoch in range(1, lastEpoch+1):
                processedFilesCount = processedFilesCount + 1
                h5DatasetName = "{}.{:03d}".format(modelsName, epoch)
                y_pred = predictions[h5DatasetName][()]
                
                if taskType == "regression":
                    # Compute MAE (Mean Absolute Error)
                    mae = Metrics.MAE(y_true, y_pred)
                    jsonData["mae"].append(mae)
                    
                    # Compute Median Absolute Error
                    median = Metrics.MedianAbsoluteError(y_true, y_pred)
                    jsonData["medianabsoluteerror"].append(median)
                    
                    # Compute MAPE (Mean Absolute Percentage Error)
                    mape = Metrics.MAPE(y_true, y_pred)
                    jsonData["mape"].append(mape)
                    
                    # Compute MSE (Mean Squared Error)
                    mse = Metrics.MSE(y_true, y_pred)
                    jsonData["mse"].append(mse)
                    
                    # Compute REC (Regression Error Characteristic) area under curve
                    recauc = Metrics.REC(y_true, y_pred, maxTolerance)
                    jsonData["recauc"].append(recauc)  
                    print("MAE: {:.3f}\tMAPE: {:.3f}\tMSE: {:.3f}\tREC-AUC: {:.3f}".format(mae, mape, mse, recauc))
                elif taskType == "classification":
                    f1 = Metrics.F1(y_true, y_pred)
                    jsonData["f1"].append(f1)
                    print("Weighted F1: {:.3f}".format(f1))
                else:
                    print("Unknown task type")
                    exit(1)
            
            if taskType == "regression":
                AUC_RECAUC = Metrics.AUCRECAUC(jsonData["recauc"])
                AUC_Loss = Metrics.AUCLoss(jsonData["mse"])
                print("AUC_RECAUC: {:.4f}".format(AUC_RECAUC))
                print("AUC_Loss: {:.4f}".format(AUC_Loss))
                jsonData["aucrecauc"] = AUC_RECAUC
                jsonData["aucloss"] = AUC_Loss
                convergence_rate = Metrics.ConvergenceRateRegression(jsonData["mae"])
                print("Convergence rate: {:.4f}".format(convergence_rate))
                jsonData["convergence_rate"] = convergence_rate
            elif taskType == "classification":
                f1auc = Metrics.F1AUC(jsonData["f1"])
                jsonData["f1auc"] = f1auc
                print("Weighted F1 AUC: {:.3f}".format(f1auc))
                convergence_rate = Metrics.ConvergenceRateClassification(jsonData["f1"])
                print("Convergence rate: {:.4f}".format(convergence_rate))
                jsonData["convergence_rate"] = convergence_rate
            if convergence_rate < 0:
                print("Error: convergence_rate < 0")
                exit(1)
            predictions.close()
            
            
            with open("{}/metrics.json".format(modelsDir), 'w') as outfile:
                json.dump(jsonData, outfile)

print(processedFilesCount)