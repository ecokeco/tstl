import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def MAE(y_true, y_pred):
    mae = np.abs(y_true - y_pred)
    mae = np.mean(mae)
    return mae
    
def MedianAbsoluteError(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    median = np.median(err)
    return median
    
def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_true = np.abs(y_true)
    mask = y_true == 0
    mask = np.logical_not(mask)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mape = np.abs(y_true - y_pred) / y_true
    mape = np.mean(mape)
    return mape
    
def MSE(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))
    
# Function for Regression Error Characteristic Curve
def REC(y_true , y_pred, End_Range, points=10000):
    # initializing the values for Epsilon
    Begin_Range = 0
    Interval_Size = End_Range/points
    
    # List of epsilons
    Epsilon = np.arange(Begin_Range , End_Range , Interval_Size)
    
    # initilizing the lists
    Accuracy = []
    
    # Compute m
    diff = np.abs(y_true - y_pred)

    # Main Loops
    for i in range(len(Epsilon)):
        count = np.sum(diff < Epsilon[i])
        Accuracy.append(count/len(y_true))
    
    # Calculating Area Under Curve using trapezoid rule
    AUC = np.trapz(Accuracy , Epsilon ) / np.max(Epsilon)
        
    # returning area under curve    
    return AUC
    
def AUCRECAUC(measuredAUCs):
    deltas = np.arange(0, len(measuredAUCs), 1)
    auc = np.trapz(measuredAUCs, deltas)
    rect_width = len(measuredAUCs) - 1
    rect_height = max(measuredAUCs)
    return auc / (rect_width * rect_height)

def ConvergenceRateRegression(measuredMAEs):
    epochs = np.arange(0, len(measuredMAEs), 1)
    rect_width = max(epochs)
    rect_height = max(measuredMAEs) - min(measuredMAEs)
    auc = np.trapz(measuredMAEs, epochs) - rect_width * min(measuredMAEs)
    return 1.0 - auc / (rect_width * rect_height)

def ConvergenceRateClassification(measuredF1s):
    epochs = np.arange(0, len(measuredF1s), 1)
    rect_width = max(epochs)
    rect_height = max(measuredF1s) - min(measuredF1s)
    auc = np.trapz(measuredF1s, epochs) - rect_width * min(measuredF1s)
    return auc / (rect_width * rect_height)
    
def AUCLoss(measuredLosses):
    deltas = np.arange(0, len(measuredLosses), 1)
    auc = np.trapz(measuredLosses, deltas)
    return auc/max(deltas)
    
def Accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    
def F1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")
    
def F1AUC(measuredF1s):
    deltas = np.arange(0, len(measuredF1s), 1)
    auc = np.trapz(measuredF1s, deltas)
    rect_width = len(measuredF1s) - 1
    rect_height = max(measuredF1s)
    return auc / (rect_width * rect_height)
    
# This test is applicable only for regression models
def WilcoxonSignedRankTest(referentModelPredictions, tlModelPredictions, datasetName):
    import scipy
    import scipy.stats as stats
    import os
    scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
    
    # Load ground truth values
    y_true = np.load(scriptDir + "../data/{}_test_Y.npy".format(datasetName))[:, 0]
    
    # Compute error vectors
    err1 = np.abs(y_true - referentModelPredictions)
    err2 = np.abs(y_true - tlModelPredictions)
    
    statistic, pvalue = stats.wilcoxon(err1, err2)
    return pvalue
    
# This test is applicable only for classification models
# reference: https://www.tandfonline.com/doi/abs/10.1080/01621459.1948.10483284
def BowkerTest(nbOfClasses, referentModelPredictions, tlModelPredictions):
    import statsmodels.api as sm
    
    if max(referentModelPredictions.max(), tlModelPredictions.max()) >= nbOfClasses:
        print("There are more classes than you specified")
        exit(1)
    
    contigencyTbl = np.zeros((nbOfClasses, nbOfClasses))
    for row in range(0, nbOfClasses):
        for col in range(0, nbOfClasses):
            pred1 = (referentModelPredictions == col)
            pred2 = (tlModelPredictions == row)
            overlap = np.logical_and(pred1, pred2)
            overlapCount = np.sum(overlap)
            contigencyTbl[row, col] = overlapCount
    
    table = sm.stats.SquareTable(contigencyTbl)
    values = table.symmetry()
    return values.pvalue