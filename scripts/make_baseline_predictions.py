import os
from TransferLearning import getBestModel
import tensorflow as tf
import sys
import numpy as np
import tcn
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow.keras.backend as K
from scipy.integrate import simps
from sklearn.metrics import r2_score
import Metrics
import json
import h5py

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"

memmapMode = None
models = ["convnetquakeingv", "magnet", "mlstm_fcn", "tcn"]
datasets = [
    ("lomax", True, "regression"),
    ("lendb", True, "regression"),
    ("stead", False, "regression"),
    ("speech8khz", True, "classification"),
    ("emg", True, "classification"),
    ("sp500", True, "regression")    
]


for datasetName, useStreamMax, taskType in datasets:
    # Open test dataset 
    print("Loading dataset {}".format(datasetName))
    X_test_waveforms = np.load(scriptDir + "../data/" + datasetName + "_test_X.npy", mmap_mode=memmapMode)
    Y_test = np.load(scriptDir + "../data/" + datasetName + "_test_Y.npy", mmap_mode=memmapMode)
    if useStreamMax:
        X_test_stream_max = np.load(scriptDir + "../data/" + datasetName + "_test_STREAM_MAX.npy", mmap_mode=memmapMode)
    else:
        X_test_stream_max = None
        
    for modelName in models:
        baselineModelName = modelName + "-" + datasetName
        print("Processing model {}".format(baselineModelName))
        modelFilename, epoch = getBestModel(baselineModelName, return_epoch=True)
        model = tf.keras.models.load_model(scriptDir + "../models/" + baselineModelName + "/" + modelFilename + ".h5", custom_objects={"TCN": tcn.TCN}, compile=False)
        
        if useStreamMax:
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

        # Create predictions.h5 file
        h5File = scriptDir + "../models/{}/predictions.h5".format(baselineModelName)
        if not os.path.exists(h5File):
            predictions = h5py.File(h5File, "w")
            predictions.attrs["dataset"] = datasetName
            predictions.attrs["useStreamMax"] = useStreamMax
            predictions.attrs["baselineModelName"] = baselineModelName
            print("{}: Creating".format(baselineModelName))
        else:
            predictions = h5py.File(h5File, "r+")
            print("{}: Already exists".format(baselineModelName))
        predictions.create_dataset(baselineModelName, data=y_pred)
        predictions.close()
