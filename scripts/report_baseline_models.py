import os
import h5py as h5
import sys
import numpy as np
import tcn
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow.keras.backend as K
from scipy.integrate import simps
from sklearn.metrics import r2_score
import Metrics
import json

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
report = open(scriptDir + "../reports/baseline_models.html", "w")

memmapMode = None
models = ["convnetquakeingv", "magnet", "mlstm_fcn", "tcn"]
datasets = [
    ("lomax", 4.3, "regression"),
    ("lendb", 4.3, "regression"),
    ("stead", 4.3, "regression"),
    ("speech8khz", None, "classification"),
    ("emg", None, "classification"),
    ("sp500", 1960, "regression")    
]

MAEs = {}
MedianAbsoluteErrors = {}
MAPEs = {}
F1s = {}

for datasetName, endRange, problemType in datasets:
    # Open test dataset 
    print("Loading dataset {}".format(datasetName))
    Y_test = np.load(scriptDir + "../data/" + datasetName + "_test_Y.npy", mmap_mode=memmapMode)
        
    for modelName in models:
        baselineModelName = modelName + "-" + datasetName
        print("Processing model {}".format(baselineModelName))
        
        h5file = h5.File(scriptDir + "../models/{}/predictions.h5".format(baselineModelName))
        y_pred = h5file[baselineModelName][()]
        
        if problemType == "regression":
            y_true = Y_test[:, 0]
        else:
            y_true = Y_test.argmax(axis=-1)

        if problemType == "regression":
            MAEs[baselineModelName] = Metrics.MAPE(y_true, y_pred)
            MAPEs[baselineModelName] = Metrics.MAPE(y_true, y_pred)
            MedianAbsoluteErrors[baselineModelName] = Metrics.MedianAbsoluteError(y_true, y_pred)
        elif problemType == "classification":
            F1s[baselineModelName] = Metrics.F1(y_true, y_pred)
        else:
            print("Unknown problem type")
            exit(1)
        
        
print("Writing to file")
report.write("<html><body>")

# Generate MAE table
report.write("<br><h2>Mean absolute error</h2>")
report.write("<table border=1><tr><td></td>")
for datasetName, endRange, problemType in datasets:
    report.write("<td><b>{}</b></td>".format(datasetName))
report.write("</tr>")

for modelName in models:
    report.write("<tr>")
    report.write("<td><b>{}</b></td>".format(modelName))
    for datasetName, endRange, problemType in datasets:
        baselineModelName = modelName + "-" + datasetName
        if baselineModelName in MAEs:
            report.write("<td>{:.4f}</td>".format(MAEs[baselineModelName]))
        else:
            report.write("<td></td>")
    report.write("</tr>")
report.write("</table>")


# Generate Median Absolute Error table
report.write("<br><h2>Median absolute error</h2>")
report.write("<table border=1><tr><td></td>")
for datasetName, endRange, problemType in datasets:
    report.write("<td><b>{}</b></td>".format(datasetName))
report.write("</tr>")

for modelName in models:
    report.write("<tr>")
    report.write("<td><b>{}</b></td>".format(modelName))
    for datasetName, endRange, problemType in datasets:
        baselineModelName = modelName + "-" + datasetName
        if baselineModelName in MedianAbsoluteErrors:
            report.write("<td>{:.4f}</td>".format(MedianAbsoluteErrors[baselineModelName]))
        else:
            report.write("<td></td>")
    report.write("</tr>")
report.write("</table>")


# Generate MAPE table
report.write("<br><h2>Mean absolute percentage error</h2>")
report.write("<table border=1><tr><td></td>")
for datasetName, endRange, problemType in datasets:
    report.write("<td><b>{}</b></td>".format(datasetName))
report.write("</tr>")

for modelName in models:
    report.write("<tr>")
    report.write("<td><b>{}</b></td>".format(modelName))
    for datasetName, endRange, problemType in datasets:
        baselineModelName = modelName + "-" + datasetName
        if baselineModelName in MAPEs:
            report.write("<td>{:.2f}</td>".format(MAPEs[baselineModelName] * 100.0))
        else:
            report.write("<td></td>")
    report.write("</tr>")
report.write("</table>")


# Generate Weighted F1 score table
report.write("<br><h2>Weighted F1 score</h2>")
report.write("<table border=1><tr><td></td>")
for datasetName, endRange, problemType in datasets:
    report.write("<td><b>{}</b></td>".format(datasetName))
report.write("</tr>")

for modelName in models:
    report.write("<tr>")
    report.write("<td><b>{}</b></td>".format(modelName))
    for datasetName, endRange, problemType in datasets:
        baselineModelName = modelName + "-" + datasetName
        if baselineModelName in F1s:
            report.write("<td>{:.2f}%</td>".format(F1s[baselineModelName] * 100.0))
        else:
            report.write("<td></td>")
    report.write("</tr>")
report.write("</table>")

report.flush()
report.close()