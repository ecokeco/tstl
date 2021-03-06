import time
import tensorflow.keras as keras
from tensorflow.keras import backend as K

"""
Taken from https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
"""

class CustomHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.learningRates = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.learningRates.append(K.eval(self.model.optimizer.lr))