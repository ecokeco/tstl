#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["TF_KERAS"] = "1"

import sys
import os
from os import listdir, walk
from os.path import isfile, join, isdir
import numpy as np
import shutil
import random
from datetime import datetime
from datetime import timedelta
import os.path
import argparse



scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
if os.path.exists(scriptDir + "../models") == False:
    os.mkdir(scriptDir + "../models")


parser = argparse.ArgumentParser()
parser.add_argument("model", help="which model to train")
parser.add_argument("dataset", help="which dataset to use")
parser.add_argument("type", help="Type of problem. Can be classification or regression.", default="regression")
parser.add_argument("--use-max-stream", help="should model use stream max.", action='store_true', default=False)
parser.add_argument("--lr", help="specifies learning rate.", default=0.001)
parser.add_argument("--epochs", help="specifies number of epochs", default=150, type=int)
parser.add_argument("--batch-size", help="specifies learning rate", default=32, type=int)
parser.add_argument("--cudnn-lstm", help="specify in order to use CuDNNLSTM implementation instead. This implementation is faster.", default=False, action="store_true")
parser.add_argument("--memmap", help="use this option if you don't have enough RAM to load all data at once.", default=False, action="store_true")
parser.add_argument("--save-best", help="saves best model along with latest model.", default=False, action="store_true")
parser.add_argument("--save-all", help="saves model each epoch", default=False, action="store_true")
parser.add_argument("--lr-reducer", help="reduces learning rate by a factor of 5 once learning stagnates.", default=False, action="store_true")
parser.add_argument("--early-stopping", help="Turns on early stopping. Training will be stopped if no progress is made within 10 epochs.", default=False, action="store_true")
parser.add_argument("--summary-only", help="prints model summary and then exits", action='store_true', default=False)
parser.add_argument("--name", help="Specifies under which name is model saved. Model won't be saved if name is not specified.", default=None)
parser.add_argument("--weights", help="Specifies which model's weights to load.", default=None)
parser.add_argument("--freeze-layers", help="Freezes layers during transfer learning.", action='store_true', default=False)
parser.add_argument("--keep-all-weights", help="", action='store_true', default=False)
parser.add_argument("--lr-multiplier", help="", default=1.0, type=float)
parser.add_argument("--gpu", help="Specifies which GPU to use.", default=None)
parser.add_argument("--verbose", help="Specifies verbose level to use. Default is 1.", default=1, type=int)
parser.add_argument("--log-to-file", help="Redirects STDOUT and STDERR to files inside model's directory.", action='store_true', default=False)
args = parser.parse_args()

# Create directory for model
if args.name is not None:
    modelsDir = scriptDir + "../models/" + args.name + "/"
    os.mkdir(modelsDir)

# Create log file if specified
if args.log_to_file:
    if args.name is None:
        print("You must specify model's name in order to use --log-to-file")
        exit(1)
    else:
        stdoutFile = open(scriptDir + "../models/{}/stdout.log".format(args.name), "w")
        stderrFile = open(scriptDir + "../models/{}/stderr.log".format(args.name), "w")
        sys.stdout = stdoutFile
        sys.stderr = stderrFile

from tensorflow.keras.layers import add, ConvLSTM2D, Reshape, Dense, AveragePooling2D, Input, Conv2DTranspose, TimeDistributed, Dropout, Flatten, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import add, Reshape, Dense, Input, TimeDistributed, Dropout, Activation, LSTM, Conv1D, Cropping1D
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf
import tcn
from keras_lr_multiplier import LRMultiplier
from DataGenerator import DataGenerator
from CustomHistory import CustomHistory


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

# Open specified training dataset
X_train_waveforms = np.load(scriptDir + "../data/" + args.dataset + "_train_X.npy", mmap_mode=memmapMode)
Y_train = np.load(scriptDir + "../data/" + args.dataset + "_train_Y.npy", mmap_mode=memmapMode)
if args.use_max_stream:
    X_train_stream_max = np.load(scriptDir + "../data/" + args.dataset + "_train_STREAM_MAX.npy", mmap_mode=memmapMode)
else:
    X_train_stream_max = None

# Open specified validation dataset 
X_validation_waveforms = np.load(scriptDir + "../data/" + args.dataset + "_validation_X.npy", mmap_mode=memmapMode)
Y_validation = np.load(scriptDir + "../data/" + args.dataset + "_validation_Y.npy", mmap_mode=memmapMode)
if args.use_max_stream:
    X_validation_stream_max = np.load(scriptDir + "../data/" + args.dataset + "_validation_STREAM_MAX.npy", mmap_mode=memmapMode)
else:
    X_validation_stream_max = None
 
# Determine waveform shape
inputShapes = []
inputShapes.append(X_train_waveforms[0].shape)
if args.use_max_stream:
    inputShapes.append(X_train_stream_max[0].shape)

# Determine output size
outputSize = len(Y_train[0])

# Generators
trainingGenerator = DataGenerator(X_train_waveforms, X_train_stream_max, Y_train, batch_size=args.batch_size, shuffle=True)
validationGenerator = DataGenerator(X_validation_waveforms, X_validation_stream_max, Y_validation, batch_size=args.batch_size, shuffle=True)

# Get model
import Models
model = getattr(Models, args.model)(
    inputShapes = inputShapes,
    outputSize = outputSize,
    problemType = args.type,
    useMaxStream = args.use_max_stream,
    useCuDNNLSTM = args.cudnn_lstm
    )
if args.name is not None:
    model._name = args.name

# Load weights if specified
lrMultipliers = None
if args.weights != None:
    from TransferLearning import copyWeights, copyWeightsThatMatch, getBestModel
    filename = scriptDir + "../models/{}.h5".format(args.weights)
    if not isfile(filename):
        filename = scriptDir + "../models/{}/{}.h5".format(args.weights, getBestModel(args.weights))
    print("Loading pretrained model from {}".format(filename))
    pretrainedModel = tf.keras.models.load_model(filename, compile=False, custom_objects={"TCN": tcn.TCN})
    if args.keep_all_weights:
        lrMultipliers = copyWeightsThatMatch(pretrainedModel, model, freezeLayers=args.freeze_layers, lrMultiplier=args.lr_multiplier)    
    else:
        lrMultipliers = copyWeights(pretrainedModel, model, [Conv1D, tcn.TCN], freezeLayers=args.freeze_layers, lrMultiplier=args.lr_multiplier)    

# Prepare loss function, optimizer and compile model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy

if args.weights is not None:
    optimizer = LRMultiplier(Adam(lr=float(args.lr)), lrMultipliers)
else:
    optimizer = Adam(lr=float(args.lr))
lossFunction = None
metrics = None
if args.type == "regression":
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    lossFunction = mean_squared_error
elif args.type == "classification":
    metrics = ["accuracy"]
    lossFunction = categorical_crossentropy
else:
    print("Unknown problem type specified!")
    exit(1)
    
model.compile(
                optimizer=optimizer, 
                loss=lossFunction,
                metrics=metrics
                )

#Print model summary
#tensorflow.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
print(model.summary())
if args.summary_only:
    exit(0)
    
# Define callbacks
callbacks = []

# Create callback that measures training time per epoch
custom_callback = CustomHistory()
callbacks.append(custom_callback)

# Early stopping
if args.early_stopping:
    if args.type == "regression":
        early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)
    elif args.type == "classification":
        early_stopping_monitor = EarlyStopping(monitor='val_acc', patience=10)
    callbacks.append(early_stopping_monitor)

# lr reducer
if args.lr_reducer:
    if args.type == "regression":
        lr_reducer = ReduceLROnPlateau(factor=0.2,
                                cooldown=0,
                                patience=4,
                                min_lr=0.5e-6,
                                monitor="val_loss",
                                verbose=1)
    elif args.type == "classification":
        lr_reducer = ReduceLROnPlateau(factor=0.2,
                                cooldown=0,
                                patience=4,
                                min_lr=0.5e-6,
                                monitor="val_acc",
                                verbose=1)
    callbacks.append(lr_reducer)

# Callback for saving best model
if args.save_best:
    if args.name is not None:
        if args.type == "regression":
            best_save = ModelCheckpoint(modelsDir + args.name + ".best_{epoch:03d}.h5", save_best_only=True, monitor='val_loss')
        elif args.type == "classification":
            best_save = ModelCheckpoint(modelsDir + args.name + ".best_{epoch:03d}.h5", save_best_only=True, monitor='val_acc')
        callbacks.append(best_save)
    else:
        print("You specified --save-best option without specifying --name. Exiting.")
        exit(1)

# Callback for saving model each epoch
if args.save_all:
    if args.name is not None:
        save_all = ModelCheckpoint(modelsDir + args.name + ".{epoch:03d}.h5")
        callbacks.append(save_all)
    else:
        print("You specified --save-all option without specifying --name. Exiting.")
        exit(1)
    
# Fit model
history = model.fit_generator(generator=trainingGenerator, 
                    epochs=args.epochs,
                    validation_data=validationGenerator,
                    shuffle=True,
                    verbose=args.verbose,
                    callbacks=callbacks)

                    
# Save model and training history (if name is specified)
if args.name is not None:
    with open(modelsDir + "{}.history".format(args.name), "wb") as file_pi:
        h = history.history
        h["epoch_time"] = custom_callback.times
        h["epoch_lr"] = custom_callback.learningRates
        pickle.dump(h, file_pi)
    model.save(modelsDir + "{}.h5".format(args.name))
