import h5py as h5
import random
import numpy as np
import os
import argparse
import pandas as pd
from tensorflow.keras.utils import to_categorical

WINDOW_LENGTH = 80
WINDOW_STEP = 10
NB_CLASSES = 4
NB_USED_CHANNELS = 3
TRAINING_PERCENTAGE = 70
TEST_PERCENTAGE = 15
VALIDATION_PERCENTAGE = 15

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
parser = argparse.ArgumentParser()
parser.add_argument("pathToDataset", help="Path to Pinch dataset")
parser.add_argument("name", help="Output filename (name of dataset after preprocessing)")
args = parser.parse_args()

def labelToInt(label):
    if label == "Pinch1":
        return 0
    elif label == "Pinch2":
        return 1
    elif label == "Pinch3":
        return 2
    elif label == "Pinch4":
        return 3
    elif label == "none":
        return -1
    else:
        print("Unknown label {}".format(label))
        exit(1)
        

files = set()   # Set will keep single instance of each file
for filename in os.listdir(args.pathToDataset):
    if filename.endswith(".npy"):
        if filename.endswith("_ann.npy"):
            index = filename.find("_ann.npy")
        elif filename.endswith("_emg.npy"):
            index = filename.find("_emg.npy")
        else:
            print("Unknown file {}".format(filename))
            exit(0)
        filename = filename[:index]
    files.add(filename)

print("Found data for {} sessions!".format(len(files)))
    
    
# Init samples array (each class has it's own subarray)
samples = [ [] for i in range(NB_CLASSES) ]
    
print("Found {} files".format(len(files)))
totalSamples = 0
for filename in files:
    print("Processing {}".format(filename))
    data = np.load(args.pathToDataset + "/" + filename + "_emg.npy")
    labels = np.load(args.pathToDataset + "/" + filename + "_ann.npy")
    
    for i in range(0, len(data), WINDOW_STEP):
        end = i + WINDOW_LENGTH
        if end > len(data):
            break
        X = np.zeros((WINDOW_LENGTH, NB_USED_CHANNELS))
        X[:, 0] = data[i:end, 0]   # Copy NB_USED_CHANNELS input channels
        X[:, 1] = data[i:end, 2]   # Copy NB_USED_CHANNELS input channels
        X[:, 2] = data[i:end, 5]   # Copy NB_USED_CHANNELS input channels
        
        Y = None
        
        classUnique = True
        for j in range(i+1, end):
            if labels[j] != labels[j-1]:
                classUnique = False
                break
        
        if classUnique:
            Y = labelToInt(labels[i])
            if Y == -1:
                continue
            samples[Y].append(X)
            totalSamples = totalSamples + 1
    
# Print info and shuffle samples
for i in range(NB_CLASSES):
    random.shuffle(samples[i])
    print("Class {}: {}".format(i, len(samples[i])))

# Separate samples into training, validation and test set
train = []
validation = []
test = []
for i in range(NB_CLASSES):
    classSamples = samples[i]
    trainEnd = int(TRAINING_PERCENTAGE/100.0 * len(classSamples))
    validationEnd = trainEnd + int(VALIDATION_PERCENTAGE/100.0 * len(classSamples))
    for j in range(0, trainEnd):
        train.append((classSamples[j], i))
        
    for j in range(trainEnd, validationEnd):
        validation.append((classSamples[j], i))
        
    for j in range(validationEnd, len(classSamples)):
        test.append((classSamples[j], i))
        
# Print info
print("Training set: {}".format(len(train)))
print("Validation set: {}".format(len(validation)))
print("Test set: {}".format(len(test)))

# Randomly shuffle generated sets
random.shuffle(train)
random.shuffle(validation)
random.shuffle(test)

# Write training data to file
print("Writing training data")
mask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.name), mode="w+", dtype="float64", shape=(WINDOW_LENGTH, NB_USED_CHANNELS))
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), WINDOW_LENGTH, NB_USED_CHANNELS))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), NB_CLASSES))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), 1))
for index, sample in enumerate(train):
    data = sample[0]
    classId = sample[1]
    X[index, :, :] = data
    Y[index, :] = to_categorical(classId, num_classes=NB_CLASSES)
    
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


# Write validation data to file
print("Writing validation data")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), WINDOW_LENGTH, NB_USED_CHANNELS))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), NB_CLASSES))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 1))
for index, sample in enumerate(validation):
    data = sample[0]
    classId = sample[1]
    X[index, :, :] = data
    Y[index, :] = to_categorical(classId, num_classes=NB_CLASSES)
    
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


# Write test data to file
print("Writing test data")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), WINDOW_LENGTH, NB_USED_CHANNELS))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), NB_CLASSES))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 1))
for index, sample in enumerate(test):
    data = sample[0]
    classId = sample[1]
    X[index, :, :] = data
    Y[index, :] = to_categorical(classId, num_classes=NB_CLASSES)
    
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