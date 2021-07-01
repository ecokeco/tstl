import os
import sys
import numpy as np
from scipy.integrate import simps
import Metrics
import argparse
import json
import h5py

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"


models = [
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
    
]
modelNameTranslation = {
    "mlstm_fcn-lomax": "MLSTM FCN [Lomax]",
    "mlstm_fcn-lendb": "MLSTM FCN [LEN-DB]",
    "mlstm_fcn-stead": "MLSTM FCN [STEAD]",
    "mlstm_fcn-speech8khz": "MLSTM FCN [Speech]",
    "mlstm_fcn-emg": "MLSTM FCN [EMG]",
    "mlstm_fcn-sp500": "MLSTM FCN [S\\&P 500]",
    
    "tcn-lomax": "TCN [Lomax]",
    "tcn-lendb": "TCN [LEN-DB]",
    "tcn-stead": "TCN [STEAD]",
    "tcn-speech8khz": "TCN [Speech]",
    "tcn-emg": "TCN [EMG]",
    "tcn-sp500": "TCN [S\\&P 500]",
    
    "convnetquakeingv-lomax": "ConvNet [Lomax]",
    "convnetquakeingv-lendb": "ConvNet [LEN-DB]",
    "convnetquakeingv-stead": "ConvNet [STEAD]",
    "convnetquakeingv-speech8khz": "ConvNet [Speech]",
    "convnetquakeingv-emg": "ConvNet [EMG]",
    "convnetquakeingv-sp500": "ConvNet [S\\&P 500]",
    
    "magnet-lomax": "MagNet [Lomax]",
    "magnet-lendb": "MagNet [LEN-DB]",
    "magnet-stead": "MagNet [STEAD]",
    "magnet-speech8khz": "MagNet [Speech]",
    "magnet-emg": "MagNet [EMG]",
    "magnet-sp500": "MagNet [S\\&P 500]",
}
datasets = [
    ("lomax1k5", "regression"),
    ("lomax9k", "regression"),
    ("lendb1k5", "regression"),
    ("lendb9k", "regression"),
    ("stead1k5", "regression"),
    ("stead9k", "regression"),
    ("speech8khz1k5", "classification"),
    ("speech8khz9k", "classification"),
    ("emg1k5", "classification"),
    ("emg9k", "classification"),
    ("sp5001k5", "regression"),
    ("sp5009k", "regression")
]
lrMultipliers = ["2.0", "1.5", "1.25", "1.0", "0.75", "0.5", "0.25", "0.1", "0.05", "0.01"]

tlWins = 0
tlLoses = 0
for baselineModelName, ignoredDatasets in models:
    for datasetName, taskType in datasets:
        if datasetName in ignoredDatasets:
            continue
            
        # Load convergence rates of TL models
        tl_crs = []
        for run in range(7):
            performanceBest = None
            correspondingCR = None
            for lrMultiplier in lrMultipliers:
                modelName = baselineModelName + "-" + datasetName + "_" + str(run) + "-" + lrMultiplier
                # Load models metrics
                with open(scriptDir + "../models/{}/metrics.json".format(modelName)) as json_file:
                    metrics = json.load(json_file)
                bestEpoch =  metrics["bestEpoch"]
                
                if taskType == "regression":
                    metricVal = metrics["mae"][bestEpoch-1]
                    if performanceBest is None or performanceBest > metricVal:
                        performanceBest = metricVal
                        correspondingCR = metrics["convergence_rate"]
                elif taskType == "classification":
                    metricVal = metrics["f1"][bestEpoch-1]
                    if performanceBest is None or performanceBest < metricVal:
                        performanceBest = metricVal
                        correspondingCR = metrics["convergence_rate"]
            tl_crs.append(correspondingCR) 
        
        # Load convergence rates of Non-TL models
        nontl_crs = []
        for run in range(7):
            modelName = baselineModelName.split("-")[0] + "-" + datasetName + "_" + str(run)
            
            # Load models metrics
            with open(scriptDir + "../models/{}/metrics.json".format(modelName)) as json_file:
                metrics = json.load(json_file)
            
            nontl_crs.append(metrics["convergence_rate"])
        
        tl_cr_avg = sum(tl_crs)/len(tl_crs)
        nontl_cr_avg = sum(nontl_crs)/len(nontl_crs)
        print("{} -> {} ({}): {:.2f} {:.2f} {:.2f} {:.2f}".format(baselineModelName, datasetName, taskType, tl_cr_avg, min(tl_crs), max(tl_crs), nontl_cr_avg))        
        
        # Compare them by convergence rate
        diff = (tl_cr_avg - nontl_cr_avg) / nontl_cr_avg * 100
        if diff > 0:
            tlWins += 1
        elif diff < 0:
            tlLoses += 1

print("TL wins: {}".format(tlWins))
print("TL loses: {}".format(tlLoses))
print("Total: {}".format(tlWins + tlLoses))
