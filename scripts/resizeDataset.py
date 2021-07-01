import os
import sys
import argparse
import random
from scipy.io import wavfile
import numpy as np
import datetime
import shutil

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"

parser = argparse.ArgumentParser()
parser.add_argument("datasetName", help="")
parser.add_argument("newDatasetName", help="")
parser.add_argument("trainingSize", help="", type=int)
parser.add_argument("validationSize", help="", type=int)
parser.add_argument("testSize", help="", type=int)
parser.add_argument("numOfChannels", help="", type=int)
args = parser.parse_args()

oldMask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.datasetName), mode="r+")

if args.trainingSize == -1:
    trainSize = None
else:
    trainSize = args.trainingSize
    
if args.validationSize == -1:
    validationSize = None
else:
    validationSize = args.validationSize

if args.testSize == -1:
    testSize = None
else:
    testSize = args.testSize


# Process training set
print("Processing training set")
oldX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.datasetName), mode="r+")
oldY = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.datasetName), mode="r+")
oldSTREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.datasetName), mode="r+")

xshape = list(oldX.shape)
yshape = oldY.shape
streammaxshape = oldSTREAM_MAX.shape

if args.numOfChannels != xshape[2]:
    if xshape[2] != 1:
        print("Input dataset has more than one channel and you requested {} channels. I don't know how to do it.".fomrat(args.numOfChannels))
    else:
        print("Copying channel")
        xshape[2] = args.numOfChannels

if trainSize is None:
    trainSize = xshape[0]
print("Training set: {}".format(trainSize))

X = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(trainSize, xshape[1], xshape[2]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(trainSize, yshape[1]))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(trainSize, streammaxshape[1]))
mask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(xshape[1], xshape[2]))
mask[:, :] = 0

# Compute new mask
indices = np.arange(xshape[0])
np.random.shuffle(indices)
indices = indices[:trainSize]
for i in indices:
    sample = np.array(oldX[i, :, :])
    if args.numOfChannels != sample.shape[1]:
        sample = np.repeat(sample, args.numOfChannels, 1)
    
    sample[:, :] = sample[:, :] + oldMask[:, :]
    mask[:, :] = mask[:, :] + sample[:, :]
mask[:, :] = mask[:, :] / float(trainSize)

# Apply mask to training set
for index, i in enumerate(indices):
    sample = np.array(oldX[i, :, :])
    sample = sample + oldMask
    
    if args.numOfChannels != sample.shape[1]:
        sample = np.repeat(sample, args.numOfChannels, 1)
    sample = sample - mask
    X[index, :, :] = sample
    Y[index, :] = oldY[i, :]
    STREAM_MAX[index, :] = oldSTREAM_MAX[i, :]

del trainSize
del X
del Y
del STREAM_MAX
del oldX
del oldY
del oldSTREAM_MAX



# Process validation set
print("Processing validation set")
oldX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.datasetName), mode="r+")
oldY = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.datasetName), mode="r+")
oldSTREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.datasetName), mode="r+")

xshape = list(oldX.shape)
yshape = oldY.shape
streammaxshape = oldSTREAM_MAX.shape

if args.numOfChannels != xshape[2]:
    if xshape[2] != 1:
        print("Input dataset has more than one channel and you requested {} channels. I don't know how to do it.".fomrat(args.numOfChannels))
    else:
        xshape[2] = args.numOfChannels

if validationSize is None:
    validationSize = xshape[0]
print("Validation set: {}".format(validationSize))

X = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(validationSize, xshape[1], xshape[2]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(validationSize, yshape[1]))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(validationSize, streammaxshape[1]))

# Apply mask to validation set
indices = np.arange(xshape[0])
np.random.shuffle(indices)
indices = indices[:validationSize]
for index, i in enumerate(indices):
    sample = np.array(oldX[i, :, :])
    sample = sample + oldMask
    
    if args.numOfChannels != sample.shape[1]:
        sample = np.repeat(sample, args.numOfChannels, 1)
    sample = sample - mask
    X[index, :, :] = sample
    Y[index, :] = oldY[i, :]
    STREAM_MAX[index, :] = oldSTREAM_MAX[i, :]

del validationSize
del X
del Y
del STREAM_MAX
del oldX
del oldY
del oldSTREAM_MAX



# Process test set
print("Processing test set")
oldX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.datasetName), mode="r+")
oldY = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.datasetName), mode="r+")
oldSTREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.datasetName), mode="r+")

xshape = list(oldX.shape)
yshape = oldY.shape
streammaxshape = oldSTREAM_MAX.shape

if args.numOfChannels != xshape[2]:
    if xshape[2] != 1:
        print("Input dataset has more than one channel and you requested {} channels. I don't know how to do it.".fomrat(args.numOfChannels))
    else:
        xshape[2] = args.numOfChannels

if testSize is None:
    testSize = xshape[0]
print("Test set: {}".format(testSize))

X = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(testSize, xshape[1], xshape[2]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(testSize, yshape[1]))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.newDatasetName), mode="w+", dtype="float64", shape=(testSize, streammaxshape[1]))

# Apply mask to test set
indices = np.arange(xshape[0])
np.random.shuffle(indices)
indices = indices[:testSize]
for index, i in enumerate(indices):
    sample = np.array(oldX[i, :, :])
    sample = sample + oldMask
    
    if args.numOfChannels != sample.shape[1]:
        sample = np.repeat(sample, args.numOfChannels, 1)
    sample = sample - mask
    X[index, :, :] = sample
    Y[index, :] = oldY[i, :]
    STREAM_MAX[index, :] = oldSTREAM_MAX[i, :]

del testSize
del X
del Y
del STREAM_MAX
del oldX
del oldY
del oldSTREAM_MAX
del mask
del oldMask