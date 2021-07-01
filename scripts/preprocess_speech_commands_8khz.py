import os
import sys
import argparse
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import shutil
import librosa

scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
TRAINING_PERCENTAGE = 70
TEST_PERCENTAGE = 15
VALIDATION_PERCENTAGE = 15

parser = argparse.ArgumentParser()
parser.add_argument("pathToDataset", help="Path to directory that contains dataset")
parser.add_argument("name", help="Output filename (name of dataset after preprocessing)")
parser.add_argument("--size", help="Processes only a subset of specified size of original dataset.", default=None, type=int)
args = parser.parse_args()

dirs = [
            "backward", "bed", "bird", "cat", "dog",
            "down", "eight", "five", "follow", "forward",
            "four", "go", "happy", "house", "learn",
            "left", "marvin", "nine", "no", "off",
            "on", "one", "right", "seven", "sheila",
            "six", "stop", "three", "tree", "two",
            "up", "visual", "wow", "yes", "zero"
            ]

# Check dataset structure
print("Checking dataset structure...", end="")
for d in dirs:
    path = args.pathToDataset + "/" + d
    if os.path.exists(path) == False:
        print("FAILED")
        print("Path {} doesn't exist!".format(path))
        exit(1)
path = args.pathToDataset + "/_background_noise_"
if os.path.exists(path) == False:
    print("FAILED")
    print("Path {} doesn't exist!".format(path))
    exit(1)
print("OK")

# Load all noise files into single array
noise = None
for f in os.listdir(args.pathToDataset + "/_background_noise_/"):
    if f.endswith(".wav"):
        data, samplerate = librosa.load(args.pathToDataset + "/_background_noise_/" + f, sr=8000)
        if noise is None:
            noise = data
        else:
            noise = np.concatenate((noise, data))

# Construct dictionary in which we will load all data samples
classes = {}
for d in dirs:
    classes[d] = []

# Assign numerical index to each class
classIndexes = {}
for c in classes:
    print("{} -> {}".format(c, len(classIndexes)))
    classIndexes[c] = len(classIndexes)

# Prepare temporary folder
tmpDir = scriptDir + "../data/{}_tmp/".format(args.name)
os.mkdir(tmpDir)

# Process each file
fileId = 0
for d in dirs:
    print("Loading files from {}".format(d))
    for f in os.listdir(args.pathToDataset + "/" + d + "/"):
        if f.endswith(".wav"):
            data, samplerate = librosa.load(args.pathToDataset + "/" + d + "/" + f, sr=8000)
            
            if len(data) > 8000:
                print("Audio is longer than 8000 samples. Keeping first 8000 samples.")
                data = data[0:8000]
            elif len(data) < 8000:
                print("Audio is shorter than 8000 samples. Appending noise to end.")
                noiseToAppend = np.array(noise[0:(8000 - len(data))])
                noise = np.concatenate((noise[(8000 - len(data)):], noise[:(8000 - len(data))]))       # Rotates noise array
                data = np.concatenate((data, noiseToAppend))
                
            if len(data) != 8000:
                print(data.shape)
                print(data)
                print("Something went wrong. Check your code.")
                exit(-1)
            
            np.save(tmpDir + str(fileId), data)
            
            classes[d].append("{}.npy".format(fileId))
            fileId = fileId + 1
    
# Shuffle all samples inside classes
for c in classes:
    random.shuffle(classes[c])
    if args.size is not None:
        end = args.size/fileId * len(classes[c])
        classes[c] = classes[c][:end]

# Separate data into train, validation, test
train = []
validation = []
test = []
for c in classes:
    trainEnd = int(TRAINING_PERCENTAGE/100.0 * len(classes[c]))
    validationEnd = trainEnd + int(VALIDATION_PERCENTAGE/100.0 * len(classes[c]))
    
    for filename in classes[c][:trainEnd]:
        classIndex = classIndexes[c]
        train.append((filename, classIndex))
    for filename in classes[c][trainEnd:validationEnd]:
        classIndex = classIndexes[c]
        validation.append((filename, classIndex))
    for filename in classes[c][validationEnd:]:
        classIndex = classIndexes[c]
        test.append((filename, classIndex))
    
random.shuffle(train)
random.shuffle(validation)
random.shuffle(test)

# Print info
print("Total files: {}".format(len(train) + len(validation) + len(test))) 
print("Training set size: {}".format(len(train)))
print("Validation set size: {}".format(len(validation)))
print("Test set size: {}".format(len(test)))


# Generate training data
print("Writing training data")
print("0/" + str(len(train)), end="")
mask = np.lib.format.open_memmap(scriptDir + "/../data/{}_MASK.npy".format(args.name), mode="w+", dtype="float64", shape=(8000, 1))
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), 8000, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), len(classes)))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_train_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(train), 1))
for index, sample in enumerate(train):
    if index % 100 == 0:
        print("\r" + str(index) + "/" + str(len(train)), end="")
    filename = sample[0]
    classIndex = sample[1]
    data = np.load(tmpDir + filename)
    if len(data) != 8000:
        print("File {} has length of {} but expected 441000".format(filename, len(data)))
        exit(1)

    X[index, :, 0] = data
    Y[index, :] = to_categorical(classIndex, num_classes=len(classes))
    
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
print("\r" + str(len(train)) + "/" + str(len(train)))
mask[:, :] = mask[:, :] / len(X)

# Apply mask to training set
for index, key in enumerate(X):
    X[index, :, :] = X[index, :, :] - mask
    
del X
del Y
del STREAM_MAX



# Generate validation data
print("Writing validation data")
print("0/" + str(len(validation)), end="")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 8000, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), len(classes)))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_validation_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(validation), 1))
for index, sample in enumerate(validation):
    if index % 100 == 0:
        print("\r" + str(index) + "/" + str(len(validation)), end="")
    filename = sample[0]
    classIndex = sample[1]
    data = np.load(tmpDir + filename)
    if len(data) != 8000:
        print("File {} has length of {} but expected 8000".format(filename, len(data)))
        exit(1)

    X[index, :, 0] = data
    Y[index, :] = to_categorical(classIndex, num_classes=len(classes))
    
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
print("\r" + str(len(validation)) + "/" + str(len(validation)))

del X
del Y
del STREAM_MAX


# Generate test data
print("Writing test data")
print("0/" + str(len(test)), end="")
X = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_X.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 8000, 1))
Y = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_Y.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), len(classes)))
STREAM_MAX = np.lib.format.open_memmap(scriptDir + "/../data/{}_test_STREAM_MAX.npy".format(args.name), mode="w+", dtype="float64", shape=(len(test), 1))
for index, sample in enumerate(test):
    if index % 100 == 0:
        print("\r" + str(index) + "/" + str(len(test)), end="")
    filename = sample[0]
    classIndex = sample[1]
    data = np.load(tmpDir + filename)
    if len(data) != 8000:
        print("File {} has length of {} but expected 8000".format(filename, len(data)))
        exit(1)

    X[index, :, 0] = data
    Y[index, :] = to_categorical(classIndex, num_classes=len(classes))
    
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
print("\r" + str(len(test)) + "/" + str(len(test)))

del X
del Y
del STREAM_MAX
del mask

shutil.rmtree(tmpDir)