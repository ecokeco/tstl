import h5py as h5
import random
import numpy as np
import os
import argparse
import pickle5 as pickle

TRAINING_PERCENTAGE = 70
TEST_PERCENTAGE = 15
VALIDATION_PERCENTAGE = 15

# Returns true when lists have items in common
def checkIntersection(a, b): 
    return not set(a).isdisjoint(b)

parser = argparse.ArgumentParser()
parser.add_argument("pathToDataset", help="Path to hdf5 file that contains dataset")
parser.add_argument("name", help="Output filename (name of dataset after preprocessing)")
parser.add_argument("--size", help="Processes only a subset of specified size of original dataset.", default=None, type=int)
args = parser.parse_args()
    
print("Loading keys")
dataset = h5.File(args.pathToDataset, "r")
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Generate list of all earthquakes in dataset
keysUnfiltered = []
keysUnfiltered = keysUnfiltered + ["train/events/" + key for key in list(dataset["train/events"].keys())]
keysUnfiltered = keysUnfiltered + ["test/events/" + key for key in list(dataset["test/events"].keys())]
keysUnfiltered = keysUnfiltered + ["test2/events/" + key for key in list(dataset["test2/events"].keys())]
keysUnfiltered = keysUnfiltered + ["validate/events/" + key for key in list(dataset["validate/events"].keys())]

# Lomax dataset contains some duplicates so we filter them out. Unique ones are stored in keys.
keys = []
usedKeys = set()
for key in keysUnfiltered:
    datasetName = key.split("/")[-1]
    if datasetName in usedKeys:
        continue
    else:
        usedKeys.add(datasetName)
        keys.append(key)
    
print("Unfiltered keys: " + str(len(keysUnfiltered)))
print("Filtered keys: " + str(len(keys)))

print("Shuffling keys")
random.shuffle(keys)
if args.size is not None:
	keys = keys[:args.size]
n = len(keys)
signalShape = dataset[keys[0]].shape
print("Signal shape: " + str(signalShape))
print("Signals count: " + str(n))

print("Creating training dataset")
trainStart = 0
trainEnd = trainStart + int(TRAINING_PERCENTAGE/100.0 * n)
train = keys[trainStart:trainEnd]

print("Creating test dataset")
testStart = trainEnd
testEnd = testStart + int(TEST_PERCENTAGE/100.0 * n)
test = keys[testStart:testEnd]

print("Creating validation dataset")
validationStart = testEnd
validationEnd = n
validation = keys[validationStart:validationEnd]

# Check for overlapping between lists just to be sure.
print("Checking for overlapping...")
overlapping = (checkIntersection(train, test) or 
               checkIntersection(train, validation) or
               checkIntersection(test, validation))
                 
if overlapping == True:
    print("Check your code. There should be no overlapping between datasets.")
    exit(1)
else:
    print("Check OK")

# Write training data to disk
print("Writing training data")
print("0/{}".format(str(len(train))), end="")
mask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.name), mode="w+", dtype="float64", shape=(signalShape[0], signalShape[1]))
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.name), dtype='float64', mode='w+', shape=(len(train), signalShape[0], signalShape[1]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.name), dtype='float64', mode='w+', shape=(len(train), 1))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.name), dtype='float64', mode='w+', shape=(len(train), 1))
for index, key in enumerate(train):
    if index % 100 == 0:
        print("\r{}/{}".format(str(index), str(len(train))), end="")
    data = dataset[key]
    magnitude = float(data.attrs["magnitude"])

    X[index, :, :] = np.array(data)
    Y[index, 0] = magnitude
    
    # Scale to range [0, 1]
    minVal = np.min(X[index, :, :])
    maxVal = np.max(X[index, :, :])
    STREAM_MAX[index, 0] = max(abs(maxVal), abs(minVal))
    if(maxVal - minVal == 0):
        # This waveform is constant signal so it does not contain any information.
        X[index, :, :] = 0
    else:
        X[index, :, :] = X[index, :, :] - minVal
        X[index, :, :] = X[index, :, :] / (maxVal - minVal)
    mask[:, :] = mask[:, :] + X[index, :, :]
print("\r{}/{}".format(str(len(train)), str(len(train))))
mask[:, :] = mask[:, :] / len(X)

# Apply mask to training set
for index, key in enumerate(X):
    X[index, :, :] = X[index, :, :] - mask
    
del X
del Y
del STREAM_MAX


# Write test data to disk
print("Writing test data")
print("0/{}".format(str(len(test))), end="")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), signalShape[0], signalShape[1]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 1))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 1))
for index, key in enumerate(test):
    if index % 100 == 0:
        print("\r{}/{}".format(str(index), str(len(test))), end="")
    data = dataset[key]
    magnitude = float(data.attrs["magnitude"])

    X[index, :, :] = np.array(data)
    Y[index, 0] = magnitude
    
    # Scale data to range [0, 1]
    minVal = np.min(X[index, :, :])
    maxVal = np.max(X[index, :, :])
    STREAM_MAX[index, 0] = max(abs(maxVal), abs(minVal))
    if(maxVal - minVal == 0):
        # This waveform is constant signal so it does not contain any information.
        X[index, :, :] = 0
    else:
        X[index, :, :] = X[index, :, :] - minVal
        X[index, :, :] = X[index, :, :] / (maxVal - minVal)
    X[index, :, :] = X[index, :, :] - mask
print("\r{}/{}".format(str(len(test)), str(len(test))))
del X
del Y
del STREAM_MAX

# Write validation data to disk
print("Writing validation data")
print("0/{}".format(str(len(validation))), end="")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), signalShape[0], signalShape[1]))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 1))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 1))
for index, key in enumerate(validation):
    if index % 100 == 0:
        print("\r{}/{}".format(str(len(validation)), str(len(validation))), end="")
    data = dataset[key]
    magnitude = float(data.attrs["magnitude"])

    X[index, :, :] = np.array(data)
    Y[index, 0] = magnitude
    
    # Scale data to range [-1, 1]
    minVal = np.min(X[index, :, :])
    maxVal = np.max(X[index, :, :])
    STREAM_MAX[index, 0] = max(abs(maxVal), abs(minVal))
    if(maxVal - minVal == 0):
        # This waveform is constant signal so it does not contain any information.
        X[index, :, :] = 0
    else:
        X[index, :, :] = X[index, :, :] - minVal
        X[index, :, :] = X[index, :, :] / (maxVal - minVal)
    X[index, :, :] = X[index, :, :] - mask
print("\r{}/{}".format(str(len(validation)), str(len(validation))))
del X
del Y
del STREAM_MAX
del mask