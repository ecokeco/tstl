import sys
import os
os.environ["TF_KERAS"] = "1"

# Loading Packages
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn import datasets
from scipy.integrate import simps
from sklearn.metrics import r2_score

import os
import sys
import argparse
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras import backend as K
from DataGenerator import DataGenerator
from TransferLearning import getLastEpoch
import keras_lr_multiplier
from tcn import TCN

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

targetDatasets = [              # Target datasets and whether to use stream max or not
    ("stead1k5", False, "regression"),
    ("stead9k", False, "regression"),
    ("lomax1k5", True, "regression"),
    ("lomax9k", True, "regression"),
    ("lendb1k5", True, "regression"),
    ("lendb9k", True, "regression"),
    
    ("speech8khz1k5", True, "classification"),
    ("speech8khz9k", True, "classification"),
    ("emg1k5", True, "classification"),
    ("emg9k", True, "classification"),
    ("sp5001k5", True, "regression"),
    ("sp5009k", True, "regression"),
]

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
parser = argparse.ArgumentParser()
parser.add_argument("runNumber", help="")
parser.add_argument("--gpu", help="Specifies which GPU to use.", default=None)
parser.add_argument("--model-filter", help="", default=None)
parser.add_argument("--dataset-filter", help="", default=None)
parser.add_argument("--memmap", help="use this option if you don't have enough RAM to load all data at once.", default=False, action="store_true")
args = parser.parse_args()

# Perform target datasets filtration
if args.dataset_filter is not None:
    lst = targetDatasets
    targetDatasets = []
    for dataset, useStreamMax, taskType in lst:
        if args.dataset_filter in dataset:
            targetDatasets.append((dataset, useStreamMax, taskType))
print(targetDatasets)

# Perform baseline models filtration
if args.model_filter is not None:
    lst = baselineModelNames
    baselineModelNames = []
    for modelName, datasets in lst:
        if args.model_filter in modelName:
            baselineModelNames.append((modelName, datasets))
print(baselineModelNames)

# Set LR multipliers
lrMultipliers = ["0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"]
print("LR multipliers: {}".format(lrMultipliers))

# Set which GPU to use
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

# Determine memmap mode
if args.memmap:
    memmapMode = "r+"
else:
    memmapMode = None

# Compute metrics    
processedFilesCount = 0
modelsToTest = []
for baseDatasetName, useMaxStream, taskType in targetDatasets:
    dataset = "{}_{}".format(baseDatasetName, args.runNumber)
    print("Loading dataset {}".format(dataset))
    # Open specified test dataset
    X_test_waveforms = np.load(scriptDir + "../data/" + dataset + "_test_X.npy", mmap_mode=memmapMode)
    Y_test = np.load(scriptDir + "../data/" + dataset + "_test_Y.npy", mmap_mode=memmapMode)
    if useMaxStream:
        X_test_stream_max = np.load(scriptDir + "../data/" + dataset + "_test_STREAM_MAX.npy", mmap_mode=memmapMode)
    else:
        X_test_stream_max = None

    for baselineModel, ignoredDatasets in baselineModelNames:
        for lrMultiplier in lrMultipliers:
            if baseDatasetName in ignoredDatasets:
                continue
            modelsName = "{}-{}-{}".format(baselineModel, dataset, lrMultiplier)
            modelsDir = scriptDir + "../models/{}/".format(modelsName)
            if not os.path.exists(modelsDir):
                print("Model {} does not exist".format(modelsName))
            else:
                print("Processing {}".format(modelsName))
            
            lastEpoch = getLastEpoch(modelsName)
            h5File = modelsDir + "predictions.h5"
            if not os.path.exists(h5File):
                predictions = h5py.File(h5File, "w")
                predictions.attrs["dataset"] = dataset
                predictions.attrs["useMaxStream"] = useMaxStream
                predictions.attrs["lastEpoch"] = lastEpoch
                predictions.attrs["modelsName"] = modelsName
                print("{}: Creating".format(modelsName))
            else:
                predictions = h5py.File(h5File, "r+")
                print("{}: Already exists".format(modelsName))
            
            
            model = None
            for epoch in range(1, lastEpoch+1):
                modelsFilename = "{}.{:03d}.h5".format(modelsName, epoch)
                h5DatasetName = "{}.{:03d}".format(modelsName, epoch)
                
                if h5DatasetName in list(predictions.keys()):
                    continue
                
                print("Initializing model from epoch {}/{}".format(epoch, lastEpoch))
                K.clear_session()
                model = tf.keras.models.load_model(modelsDir + modelsFilename, compile=False, custom_objects={"TCN": TCN, "LRMultiplier": keras_lr_multiplier.LRMultiplier})
                
                if useMaxStream:
                    y_pred = model.predict([X_test_waveforms, X_test_stream_max], batch_size=128)
                else:
                    y_pred = model.predict([X_test_waveforms], batch_size=128)
                    
                if taskType == "regression":
                    y_pred = y_pred[:, 0]
                elif taskType == "classification":
                    y_pred = y_pred.argmax(axis=-1)
                else:
                    print("Unknown task type!")
                    exit(0)
                
                predictions.create_dataset(h5DatasetName, data=y_pred)
                processedFilesCount = processedFilesCount + 1
                print("Processed files: {}".format(processedFilesCount))
                
            
            predictions.close()

