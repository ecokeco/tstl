import os
import sys
import numpy as np
from scipy.integrate import simps
import Metrics
import argparse
import json
import h5py
import matplotlib.pyplot as plt
from statistics import median

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"

models = ["convnetquakeingv", "magnet", "mlstm_fcn", "tcn"]
srcDatasets = ["lomax", "lendb", "stead", "speech8khz", "emg", "sp500"]
dstDatasets = [
    ("lomax", "regression"),
    ("lendb", "regression"),
    ("stead", "regression"),
    ("speech8khz", "classification"),
    ("emg", "classification"),
    ("sp500", "regression")
]
modelNames = {
    "convnetquakeingv": "ConvNetQuake INGV",
    "magnet": "MagNet",
    "mlstm_fcn": "MLSTM FCN",
    "tcn": "TCN"
}

lrMultipliers = ["0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"]

meanGains = {}
for model in models:
    meanGains[model] = []
    for lr in lrMultipliers:
        gains = []
        for src in srcDatasets:
            for dstName, taskType in dstDatasets:
                if src == dstName:
                    continue
                for dstSize in ("1k5", "9k"):
                    dst = dstName + dstSize
                    for run in range(7):
                        # Load referent model's metrics
                        referentModelName = "{}-{}_{}".format(model, dst, run)
                        with open(scriptDir + "../models/{}/metrics.json".format(referentModelName)) as json_file:
                            metrics = json.load(json_file)
                        bestEpoch =  metrics["bestEpoch"]
                        
                        if taskType == "regression":
                            referentPerformance = metrics["mae"][bestEpoch-1]
                        elif taskType == "classification":
                            referentPerformance = metrics["f1"][bestEpoch-1]
                        else:
                            print("ERROR")
                            exit(1) 
                            
                        # Load TL metrics
                        tlModelName = "{}-{}-{}_{}-{}".format(model, src, dst, run, lr)
                        with open(scriptDir + "../models/{}/metrics.json".format(tlModelName)) as json_file:
                            metrics = json.load(json_file)
                        bestEpoch =  metrics["bestEpoch"]
                        
                        if taskType == "regression":
                            tlPerformance = metrics["mae"][bestEpoch-1]
                        elif taskType == "classification":
                            tlPerformance = metrics["f1"][bestEpoch-1]
                        else:
                            print("ERROR")
                            exit(1)
                            
                        # Calculate gain
                        if taskType == "regression":
                            gain = (referentPerformance - tlPerformance) / referentPerformance
                        elif taskType == "classification":
                            gain = (tlPerformance - referentPerformance) / referentPerformance
                        gains.append(gain * 100)
        meanGain = sum(gains) / len(gains)
        meanGains[model].append(meanGain)
        
# Draw plot for this combination of source and target domain
plt.clf()
plt.cla()
x = np.arange(len(lrMultipliers))
fig, ax = plt.subplots()
width = 0.1
for idx, model in enumerate(models):
    ax.bar(x + (idx - len(models)/2 + 0.5) * width, meanGains[model], width, label=modelNames[model])
ax.set_ylabel("Percentual gain in performance")    
ax.set_xlabel("Learning rate multiplier")    
ax.legend()
ax.set_xticks(x)
ax.set_xticklabels(lrMultipliers)
fig.tight_layout()
plt.grid(axis='y')
plt.savefig(scriptDir + "../reports/performance_gain_vs_lr_multipliers.png".format(src, dstName))