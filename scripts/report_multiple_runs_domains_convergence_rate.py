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
import statsmodels.stats.multitest
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
seperateTables = {}
processedFilesCount = 0
for sourceDataset in sourceDatasets:
    table[sourceDataset] = {}
    seperateTables[sourceDataset] = {}
    for targetDataset, taskType in targetDatasets: 
        if sourceDataset in targetDataset:
            continue
        print("Source dataset: {}\tTarget dataset: {}".format(sourceDataset, targetDataset))
        table[sourceDataset][targetDataset] = {"nontl": [], "tl": []}
        seperateTables[sourceDataset][targetDataset] = []
        for model in models:
            for run in range(7):
                referentModelName = "{}-{}_{}".format(model, targetDataset, run)
                #print("\tProcessing referent model {}".format(referentModelName))
        
                # Load referent model's metrics
                with open(scriptDir + "../models/{}/metrics.json".format(referentModelName)) as json_file:
                    metrics = json.load(json_file)
                
                referentCR = metrics["convergence_rate"]
                    
                # Find best TL model
                tlPerformance = None
                tlCR = None
                tlModelName_chosen = None
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
                            tlCR = metrics["convergence_rate"]
                            tlModelName_chosen = tlModelName
                    elif taskType == "classification":
                        val = metrics["f1"][bestEpoch-1]
                        if tlPerformance is None or val > tlPerformance:
                            tlPerformance = val
                            tlCR = metrics["convergence_rate"]
                            tlModelName_chosen = tlModelName
                    else:
                        print("ERROR")
                        exit(1)
                
                table[sourceDataset][targetDataset]["nontl"].append(referentCR)
                table[sourceDataset][targetDataset]["tl"].append(tlCR)
                seperateTables[sourceDataset][targetDataset].append((referentModelName, referentCR))
                seperateTables[sourceDataset][targetDataset].append((tlModelName_chosen, tlCR))

# Perform Wilcoxon signed-ranks test
pValues = []
for sourceDataset in sourceDatasets:
    for targetDataset, taskType in targetDatasets:
        if sourceDataset not in targetDataset:
            nontlCRs = table[sourceDataset][targetDataset]["nontl"]
            tlCRs = table[sourceDataset][targetDataset]["tl"]
            _, pValue = stats.wilcoxon(nontlCRs, tlCRs)
            pValues.append(pValue)
            
# Perform Two-stage Benjamini, Krieger, & Yekutieli FDR procedure
# We create a lookup table for later        
_, pValuesCorrected, _, _ = statsmodels.stats.multitest.multipletests(pValues, alpha = SIGNIFICANCE_LEVEL, method = "fdr_tsbky")
          
# Write to file
report = open(scriptDir + "../reports/domains_comparison_convergence_rate.tex", "w")
report.write("\\begin{table}[]\n")
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
testIter = 0
for sourceDataset in sourceDatasets:
    line = []
    line.append("\\textbf{" + sourceDataset + "}")
    for targetDataset, taskType in targetDatasets:
        if sourceDataset in targetDataset:
            line.append("-")
        else:
            nontlCRs = table[sourceDataset][targetDataset]["nontl"]
            tlCRs = table[sourceDataset][targetDataset]["tl"]
            
            diff = 0
            for i in range(len(nontlCRs)):
                diff += (tlCRs[i] - nontlCRs[i]) / nontlCRs[i] * 100
            diff = diff / len(nontlCRs)
            
            color = None
            if pValuesCorrected[testIter] <= SIGNIFICANCE_LEVEL:
                if diff > 0:
                    color = "green"
                else:
                    color = "red"
            testIter += 1
            
            if color is not None:
                line.append("\\cellcolor{{{}}}\\textbf{{{:.3}}}\\%".format(color, diff))
            else:
                line.append("{:.3}\\%".format(diff))
    
    report.write(" & ".join(line) + " \\\\ \\hline\n")
report.write("\\end{tabular}\n")
report.write("\\end{adjustwidth}\n")
report.write("\\end{table}\n")
    
report.flush()
report.close()