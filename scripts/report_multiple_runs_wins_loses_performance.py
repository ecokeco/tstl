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
            
        # Load performances of TL models
        tl_metric_values = []
        for run in range(0, 7):
            metricBest = None
            for lrMultiplier in lrMultipliers:
                modelName = baselineModelName + "-" + datasetName + "_" + str(run) + "-" + lrMultiplier
                # Load models metrics
                with open(scriptDir + "../models/{}/metrics.json".format(modelName)) as json_file:
                    metrics = json.load(json_file)
                bestEpoch =  metrics["bestEpoch"]
                
                if taskType == "regression":
                    metricVal = metrics["mae"][bestEpoch-1]
                    if metricBest is None or metricBest > metricVal:
                        metricBest = metricVal
                elif taskType == "classification":
                    metricVal = metrics["f1"][bestEpoch-1]
                    if metricBest is None or metricBest < metricVal:
                        metricBest = metricVal
                
            tl_metric_values.append(metricBest) 
        
        # Load performances of Non-TL models
        nontl_metric_values = []
        for run in range(0, 7):
            modelName = baselineModelName.split("-")[0] + "-" + datasetName + "_" + str(run)
            
            # Load models metrics
            with open(scriptDir + "../models/{}/metrics.json".format(modelName)) as json_file:
                metrics = json.load(json_file)
            bestEpoch =  metrics["bestEpoch"]
            
            if taskType == "regression":
                metricVal = metrics["mae"][bestEpoch-1]
            elif taskType == "classification":
                metricVal = metrics["f1"][bestEpoch-1]
            
            nontl_metric_values.append(metricVal)
        
        tl_metric_avg = sum(tl_metric_values)/len(tl_metric_values)
        nontl_metric_avg = sum(nontl_metric_values)/len(nontl_metric_values)
        print("{} -> {} ({}): {:.2f} {:.2f} {:.2f} {:.2f}".format(baselineModelName, datasetName, taskType, tl_metric_avg, min(tl_metric_values), max(tl_metric_values), nontl_metric_avg))        
        
        # Compare by performance
        if taskType == "regression":
            diff = (nontl_metric_avg - tl_metric_avg) / nontl_metric_avg * 100
            if diff > 0:
                tlWins += 1
            elif diff < 0:
                tlLoses += 1
        elif taskType == "classification":
            diff = (tl_metric_avg - nontl_metric_avg) / nontl_metric_avg * 100
            if diff > 0:
                tlWins += 1
            elif diff < 0:
                tlLoses += 1
        

print("TL wins: {}".format(tlWins))
print("TL loses: {}".format(tlLoses))
print("Total: {}".format(tlWins + tlLoses))