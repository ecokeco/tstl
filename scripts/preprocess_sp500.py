import os
import sys
import argparse
import random
from scipy.io import wavfile
import numpy as np
import pandas as pd
import datetime
import shutil

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
TRAINING_PERCENTAGE = 70
TEST_PERCENTAGE = 15
VALIDATION_PERCENTAGE = 15
WINDOW_SIZE = 50    # We use first 50 samples as input for our NN and try to predict next 1
PREDICTION_AHEAD = 0
PREDICTION_SIZE = 1
COLUMN = "Close"

parser = argparse.ArgumentParser()
parser.add_argument("pathToDataset", help="Path to csv file")
parser.add_argument("name", help="Output filename (name of dataset after preprocessing)")
parser.add_argument("--size", help="Processes only a subset of specified size of original dataset.", default=None, type=int)
args = parser.parse_args()

if not os.path.exists(args.pathToDataset):
    print("Specified file does not exist. Specified path: {}".format(args.pathToDataset))
    exit(0)

print("\rProcessing file {}".format(args.pathToDataset))
df = pd.read_csv(args.pathToDataset)
df = df[[COLUMN]]
data = df.to_numpy()
data = np.squeeze(data)

# Seperate data into train, validation and test sets
trainEnd = int(TRAINING_PERCENTAGE/100.0 * len(data))
validationEnd = trainEnd + int(VALIDATION_PERCENTAGE/100.0 * len(data))
trainData = data[:trainEnd]
validationData = data[trainEnd:validationEnd]
testData = data[validationEnd:]

# Generate training set using sliding window
train = []
for i in range(WINDOW_SIZE + PREDICTION_AHEAD + PREDICTION_SIZE, len(trainData)):
    inputWindow = trainData[i-WINDOW_SIZE-PREDICTION_AHEAD-PREDICTION_SIZE:i-PREDICTION_AHEAD-PREDICTION_SIZE]
    outputWindow = trainData[i-PREDICTION_SIZE:i]
    train.append((inputWindow, outputWindow))
    
# Generate validation set using sliding window
validation = []
for i in range(WINDOW_SIZE + PREDICTION_AHEAD + PREDICTION_SIZE, len(validationData)):
    inputWindow = validationData[i-WINDOW_SIZE-PREDICTION_AHEAD-PREDICTION_SIZE:i-PREDICTION_AHEAD-PREDICTION_SIZE]
    outputWindow = validationData[i-PREDICTION_SIZE:i]
    validation.append((inputWindow, outputWindow))

# Generate test set using sliding window
test = []
for i in range(WINDOW_SIZE + PREDICTION_AHEAD + PREDICTION_SIZE, len(testData)):
    inputWindow = testData[i-WINDOW_SIZE-PREDICTION_AHEAD-PREDICTION_SIZE:i-PREDICTION_AHEAD-PREDICTION_SIZE]
    outputWindow = testData[i-PREDICTION_SIZE:i]
    test.append((inputWindow, outputWindow))
    

# Print information
print("Training set size: {}".format(len(train)))
print("Validation set size: {}".format(len(validation)))
print("Test set size: {}".format(len(test)))

# Randomly shuffle data sets
random.shuffle(train)
random.shuffle(validation)
random.shuffle(test)
    



# Write training data
print("Writing training data")
mask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.name), mode="w+", dtype="float64", shape=(WINDOW_SIZE, 1))
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), WINDOW_SIZE, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), PREDICTION_SIZE))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), 1))
for index, data in enumerate(train):
    X[index, :, 0] = data[0]
    Y[index, :] = data[1]
    
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
mask[:, :] = mask[:, :] / len(X)

# Apply mask to training set
for index, key in enumerate(X):
    X[index, :, :] = X[index, :, :] - mask
    
del X
del Y
del STREAM_MAX


# Write validation data
print("Writing validation data")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), WINDOW_SIZE, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), PREDICTION_SIZE))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 1))
for index, data in enumerate(validation):
    X[index, :, 0] = data[0]
    Y[index, :] = data[1]
    
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
    X[index, :, :] = X[index, :, :] - mask

del X
del Y
del STREAM_MAX


# Write test data
print("Writing test data")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), WINDOW_SIZE, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), PREDICTION_SIZE))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 1))
for index, data in enumerate(test):
    X[index, :, 0] = data[0]
    Y[index, :] = data[1]
    
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
    X[index, :, :] = X[index, :, :] - mask

del X
del Y
del STREAM_MAX
del mask
