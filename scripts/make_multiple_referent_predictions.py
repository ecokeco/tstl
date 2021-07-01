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

referentModels = [
    "convnetquakeingv",
    "magnet",
    "mlstm_fcn",
    "tcn"
]

datasets = [              # Target datasets and whether to use stream max or not
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
parser.add_argument("number_of_run", help="Number of single run", type=int)
parser.add_argument("--gpu", help="Specifies which GPU to use.", default=None)
parser.add_argument("--memmap", help="use this option if you don't have enough RAM to load all data at once.", default=False, action="store_true")
args = parser.parse_args()

# Generate dataset names
lst = datasets
datasets = []
for dataset, useMaxStream, taskType in lst:
    datasets.append(("{}_{}".format(dataset, args.number_of_run), useMaxStream, taskType))
print(datasets)

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
for dataset, useMaxStream, taskType in datasets:
    print("Loading dataset {}".format(dataset))
    # Open specified test dataset
    X_test_waveforms = np.load(scriptDir + "../data/" + dataset + "_test_X.npy", mmap_mode=memmapMode)
    Y_test = np.load(scriptDir + "../data/" + dataset + "_test_Y.npy", mmap_mode=memmapMode)
    if useMaxStream:
        X_test_stream_max = np.load(scriptDir + "../data/" + dataset + "_test_STREAM_MAX.npy", mmap_mode=memmapMode)
    else:
        X_test_stream_max = None

    for referentModel in referentModels:
        modelsName = "{}-{}".format(referentModel, dataset)
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

