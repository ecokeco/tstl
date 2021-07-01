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
import scipy.stats as stats
import TransferLearning

SIGNIFICANCE_LEVEL = 0.05

models = ["convnetquakeingv", "magnet", "mlstm_fcn", "tcn"]

lrMultipliers = ["0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"]
sourceDatasets = ["lomax", "lendb", "stead", "speech8khz", "emg", "sp500"]

targetDatasets = [
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
    ("sp5009k", "regression"),
]

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"

# Compute metrics
table = {}
processedFilesCount = 0
for sourceDataset in sourceDatasets:
    table[sourceDataset] = {}
    for targetDataset, taskType in targetDatasets: 
        if sourceDataset in targetDataset:
            continue
        print("Source dataset: {}\tTarget dataset: {}".format(sourceDataset, targetDataset))
        table[sourceDataset][targetDataset] = {}
        for model in models:
            table[sourceDataset][targetDataset][model] = {"nontl": [], "tl": [], "lrmultiplier": []}
            print("")
            for run in range(7):
                referentModelName = "{}-{}_{}".format(model, targetDataset, run)
                #print("\tProcessing referent model {}".format(referentModelName))
        
                # Load referent model's metrics
                with open(scriptDir + "../models/{}/metrics.json".format(referentModelName)) as json_file:
                    metrics = json.load(json_file)
                referentConvergenceRate = metrics["convergence_rate"]
                    
                # Find best TL model
                tlPerformance = None
                tlConvergenceRate = None
                tlModelName_chosen = None
                tlLrMultiplier = None
                for lr in lrMultipliers:
                    tlModelName = "{}-{}-{}_{}-{}".format(model, sourceDataset, targetDataset, run, lr)
                    #print("\tProcessing TL model {}".format(tlModelName))
                    
                    # Load TL metrics
                    with open(scriptDir + "../models/{}/metrics.json".format(tlModelName)) as json_file:
                        metrics = json.load(json_file)
                    bestEpoch =  metrics["bestEpoch"]
                    
                    if taskType == "regression":
                        val = metrics["mae"][bestEpoch-1]
                        if tlPerformance is None or val < tlPerformance:
                            tlPerformance = val
                            tlConvergenceRate = metrics["convergence_rate"]
                            tlModelName_chosen = tlModelName
                            tlLrMultiplier = lr
                    elif taskType == "classification":
                        val = metrics["f1"][bestEpoch-1]
                        if tlPerformance is None or val > tlPerformance:
                            tlPerformance = val
                            tlConvergenceRate = metrics["convergence_rate"]
                            tlModelName_chosen = tlModelName
                            tlLrMultiplier = lr
                    else:
                        print("ERROR")
                        exit(1)
                print(tlLrMultiplier)
                table[sourceDataset][targetDataset][model]["nontl"].append(referentConvergenceRate)
                table[sourceDataset][targetDataset][model]["tl"].append(tlConvergenceRate)
                table[sourceDataset][targetDataset][model]["lrmultiplier"].append(tlLrMultiplier)
                
# Write to file
report = open(scriptDir + "../reports/convergence_rate_per_model.tex", "w")
report.write("\\begin{table}[]\n")
report.write("\\caption{Average percentual gain in convergence rate across seven reruns of the experiment for the hyperparameters that achieved the best predictive performance.}\n")
report.write("\\begin{adjustwidth}{-1in}{-1in} % adjust the L and R margins by 1 inch\n")
report.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|}\n")
report.write("\\hline\n")

# Write table header
line = []
line.append(" ") # Empty cell
for dataset, _ in targetDatasets:
    line.append("\\textbf{" + dataset + "}")
report.write(" & ".join(line) + "\\\\ \\hline\n")


# Write rows
for sourceDataset in sourceDatasets:
    for model in models:
        line = []
        line.append("{} [{}]".format(model.replace("_", "\\_"), sourceDataset))
        for targetDataset, taskType in targetDatasets:
            if sourceDataset in targetDataset:
                line.append("-")
            else:
                nonTlConvergenceRates = table[sourceDataset][targetDataset][model]["nontl"]
                tlConvergenceRates = table[sourceDataset][targetDataset][model]["tl"]
                lrMultipliers = table[sourceDataset][targetDataset][model]["lrmultiplier"]
                print(lrMultipliers)
                avgNonTl = np.mean(nonTlConvergenceRates)
                avgTl = np.mean(tlConvergenceRates)
                
                _, pValue = stats.wilcoxon(nonTlConvergenceRates, tlConvergenceRates)
                diff = 0
                for i in range(len(nonTlConvergenceRates)):
                    diff += (tlConvergenceRates[i] - nonTlConvergenceRates[i]) / nonTlConvergenceRates[i] * 100
                diff = diff / len(nonTlConvergenceRates)
                
                if pValue < SIGNIFICANCE_LEVEL:
                    if diff > 0:
                        line.append("\\cellcolor{green}" + " {:.3}\\% ({:.4})".format(diff, pValue))
                    else:
                        line.append("\\cellcolor{red}" + " {:.3}\\% ({:.4})".format(diff, pValue))
                else:
                    line.append("{:.3}\\%  ({:.4})".format(diff, pValue))
    
        report.write(" & ".join(line) + " \\\\ \\hline\n")
report.write("\\end{tabular}\n")
report.write("\\end{adjustwidth}\n")
report.write("\\end{table}\n")
    
report.flush()
report.close()
