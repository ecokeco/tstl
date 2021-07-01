from tensorflow.keras.activations import relu, linear
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, Dropout, Bidirectional, concatenate, Flatten, Multiply, GlobalAvgPool1D, Reshape, Add, MaxPool1D, Concatenate, GlobalMaxPool1D, Permute, Masking, multiply
from tensorflow.keras.models import Model
import numpy as np
import tcn

def getLastEpoch(modelName):
    import os
    import re
    scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
    modelsDir = scriptDir + "../models/{modelName}/".format(modelName=modelName)
    files = os.listdir(modelsDir)
    epochMax = -1
    for file in files:
        r = re.search("^{}.(\d+)\.h5$".format(modelName), file)
        if r is not None:
            epoch = int(r.group(1))
            if epoch > epochMax:
                epochMax = epoch
    if epochMax == -1:
        print("WARNING: Couldn't find last model for {}".format(modelName))
        return None
    else:
        return epochMax
        

def getBestModel(modelName, return_epoch=False):
    import os
    import re
    scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
    modelsDir = scriptDir + "../models/{modelName}/".format(modelName=modelName)
    files = os.listdir(modelsDir)
    epochMax = -1
    retFilename = None
    for file in files:
        r = re.search("^.*\.best_(\d+)\.h5$", file)
        if r is not None:
            epoch = int(r.group(1))
            if epoch > epochMax:
                epochMax = epoch
                retFilename = file[:-3]
    if retFilename is None:
        print("WARNING: Couldn't find best model for {}".format(modelName))
        return None
    else:
        if return_epoch:
            return (retFilename, epochMax)
        else:
            return retFilename
            
def getBestValLoss(modelName, return_epoch=False):
    import pickle
    import os
    scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"
    modelsDir = scriptDir + "../models/{modelName}/".format(modelName=modelName)
    
    filename, bestEpoch = getBestModel(modelName, return_epoch=True)
    h = pickle.load(open(modelsDir + "{}.history".format(modelName), "rb"))
    if return_epoch:
        return (h["val_loss"][bestEpoch-1], bestEpoch)
    else:
        return h["val_loss"][bestEpoch-1]
        
        

"""
Returns text after slash
Example:
    input: "test/input"
    returns: "input"
"""
def extractName(s):
    i = s.index("/")
    return s[i+1:]

def copyWeights(pretrainedModel, dstModel, typesToCopy, freezeLayers=False, lrMultiplier=1.0):
    retMultipliers = {}
    print("********************************************************* {}".format(lrMultiplier))
    for srcLayer in pretrainedModel.layers:
        # Is this type of layer that should be copied?
        shouldCopy = False
        for t in typesToCopy:
            if isinstance(srcLayer, t):
                shouldCopy = True
                break
        
        if shouldCopy:
            print("Analyzing layer {}".format(srcLayer.name))
            
            # Find matching destination layer (ValueError will be thrown if no such layer exists)
            dstLayer = dstModel.get_layer(name=srcLayer.name)
            
            # Copy weights
            weights = srcLayer.get_weights()
            if isinstance(srcLayer, Conv1D):
                retMultipliers[srcLayer.name] = lrMultiplier
                w = weights[0].copy()
                biases = weights[1].copy()
                if w.shape == dstLayer.get_weights()[0].shape:
                    print("\t\tShapes matching! Performing copy.")
                    print(w.shape)
                    print(dstLayer.get_weights()[0].shape)
                    dstLayer.set_weights([w, biases])
                    
                    dstLayer.trainable = not freezeLayers
                else:
                    if w.shape[0] != dstLayer.get_weights()[0].shape[0]:
                        print("\t\tDifferent kernel sizes! This should not happen. ABORTING")
                        print(w.shape)
                        print(dstLayer.get_weights()[0].shape)
                        exit(1)
                    elif w.shape[2] != dstLayer.get_weights()[0].shape[2]:
                        print("\t\tDifferent number of filters! This should not happen. ABORTING")
                        print(pretrainedModel.summary())
                        print(dstModel.summary())
                        exit(1)
                    elif w.shape[1] == 1 and dstLayer.get_weights()[0].shape[1] > 1:
                        dstNumChannels = dstLayer.get_weights()[0].shape[1]
                        print("\t\tCopying first channel filters to other channels.")
                        print(w.shape)
                        print(dstLayer.get_weights()[1].shape)
                        print(biases.shape)
                        w = np.repeat(w, dstNumChannels, 1)
                        
                        dstLayer.set_weights([w, biases])
                        dstLayer.trainable = not freezeLayers
                        if w.shape == dstLayer.get_weights()[0].shape:
                            print("\t\tSuccess")
                        else:
                            print("\t\tSomething went wrong. Check your code. ABORTING")
                            exit(1)
                    else:
                        print("\t\t SrcChannels:{} DstChannels:{}. This situation should not happen. ABORTING".format(srcWeights.shape[1], dstWeights.shape[1]))
                        exit(1)
            elif isinstance(srcLayer, tcn.TCN):
                retMultipliers[srcLayer.name] = lrMultiplier
                print("TCN multiplier {}".format(lrMultiplier))
                srcWeights = srcLayer.get_weights()
                dstWeights = dstLayer.get_weights()
                print("srcWeights len: {}\tdstWeights len: {}".format(len(srcWeights), len(dstWeights)))
                if len(srcWeights) != len(dstWeights):
                    print("TCN layers are different!")
                    exit(1)
                for i in range(len(srcWeights)):
                    if srcWeights[i].shape == dstWeights[i].shape:
                        print("[{}] Weights matching. Copying: {} ---> {}".format(i, srcWeights[i].shape, dstWeights[i].shape))
                        dstWeights[i] = srcWeights[i]
                    elif srcWeights[i].shape[0] != dstWeights[i].shape[0]:
                        print("\t\tDifferent kernel sizes! This should not happen. ABORTING")
                        exit(1)
                    elif srcWeights[i].shape[2] != dstWeights[i].shape[2]:
                        print("\t\tDifferent number of filters! This should not happen. ABORTING")
                        exit(1)
                    elif srcWeights[i].shape[1] == 1 and dstWeights[i].shape[1] > 1:
                        dstNumChannels = dstWeights[i].shape[1]
                        print("[{}] {} X {} ---> {}".format(i, srcWeights[i].shape, dstNumChannels, dstWeights[i].shape))
                        weights = np.repeat(srcWeights[i], dstNumChannels, 1)
                        dstWeights[i] = weights
                dstLayer.set_weights(dstWeights)
                dstLayer.trainable = not freezeLayers
                        
                
    return retMultipliers

def copyWeightsThatMatch(pretrainedModel, dstModel, freezeLayers=False, lrMultiplier=1.0):
    retMultipliers = {}
    for srcLayer in pretrainedModel.layers:
        #if counter == 1:
        #    break

        print("Analyzing layer {}".format(srcLayer.name))
        
        # Find matching destination layer (ValueError will be thrown if no such layer exists)
        dstLayer = dstModel.get_layer(name=srcLayer.name)
        if len(dstLayer.get_weights()) > 0 and dstLayer.get_weights()[0].shape == srcLayer.get_weights()[0].shape:
            print("\tCopying {}".format(srcLayer.name))
            dstLayer.set_weights(srcLayer.get_weights())
            
            dstLayer.trainable = not freezeLayers
            retMultipliers[srcLayer.name] = lrMultiplier
    return retMultipliers
        
