import numpy as np
import tensorflow.keras

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, X_stream_max, Y, batch_size, shuffle=True):
        'Initialization'
        self.X = X      # Shape: (samples, timesteps, channels)
        self.X_stream_max = X_stream_max
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.numSamples = len(X)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.numSamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numSamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def getIndexes(self):
        return self.indexes 

    def __data_generation(self, indexes):
        # Initialization
        X = np.empty((self.batch_size, self.X.shape[1], self.X.shape[2]))
        X_stream_max = np.empty((self.batch_size, 1))
        y = np.empty((self.batch_size, len(self.Y[0])))

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i, :, :] = self.X[index, :, :]

            if self.X_stream_max is not None:
                X_stream_max[i, :] = self.X_stream_max[index, :]

            # Store result
            y[i, :] = self.Y[index, :]
        
        if self.X_stream_max is None:
            return X, y
        else:
            return [X, X_stream_max], y
