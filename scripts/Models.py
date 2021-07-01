import tensorflow.keras.backend as K
from tensorflow.keras.activations import relu, linear
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, Dropout, Bidirectional, concatenate, Flatten, Multiply, GlobalAvgPool1D, Reshape, Add, MaxPool1D, Concatenate, GlobalMaxPool1D, Permute, Masking, multiply
from tensorflow.keras.models import Model
import tcn as kerasTCN

# Input to all models is in shape (batchSize, waveformLength, channels)

# TCN
def tcn(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    
    nb_filters = 24         # Parameter in original paper "Hidden"
    kernel_size = 8         # Parameter in original paper "k"
    nb_stacks = 1
    dilations = [2 ** i for i in range(9)] # Parameter in original paper "n" + 1
    dilations = kerasTCN.tcn.adjust_dilations(dilations)
    padding = 'causal'
    use_skip_connections = False
    dropout_rate = 0.0
    return_sequences = False
    activation = 'relu'
    kernel_initializer = 'he_normal'
    use_batch_norm = False
    use_layer_norm = False

    x = kerasTCN.TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            name="tcn")(waveformInput)
    
    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        x = concatenate([x, streamMaxInput])
    x = Dense(outputSize)(x)

    if problemType == "regression":
        o = Activation('linear', name='output_layer')(x)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=o)
    return model

########################################################################
#   LSTM-FCN                                                           #
#   From paper Multivariate LSTM-FCNs for Time Series Classification   #
#   https://github.com/titu1994/MLSTM-FCN                              #
########################################################################
def mlstm_fcn(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):    
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    
    MAX_SEQUENCE_LENGTH = inputShapes[0][0]
    numChannels = inputShapes[0][1]
    NUM_CELLS=8    # This was a default value

    # Creating LSTM
    lstmInput = Permute((2, 1))(waveformInput)                        # According to paper "Multivariate LSTM-FCNs for Time Series Classification"
    x = Masking()(lstmInput)
    
    # At the moment there is no support for Masking layer in combination with CuDNNLSTM so you must use LSTM
    from tensorflow.keras.layers import LSTM
    x = LSTM(NUM_CELLS, unroll=True)(x)
    
    x = Dropout(0.8)(x)

    # Creating Convolutional part
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', name="conv1d_1")(waveformInput)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform', name="conv1d_2")(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', name="conv1d_3")(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        x = concatenate([x, y, streamMaxInput])
    else:
        x = concatenate([x, y])
    x = Dense(outputSize)(x)
    
    if problemType == "regression":
        o = Activation('linear', name='output_layer')(x)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(x)

    model = Model(inputs=inputs, outputs=o)
    return model
    
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.get_shape().as_list()[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

########################
#   ConvNetQuakeINGV   #
########################
def convnetquakeingv(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    l2 = 1e-3
    
    # Convolution 1
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv1")(waveformInput) 

    # Convolution 2
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv2")(current_layer) 

    # Convolution 3
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv3")(current_layer) 

    # Convolution 4
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv4")(current_layer) 

    # Convolution 5
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv5")(current_layer) 

    # Convolution 6
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv6")(current_layer) 

    # Convolution 7
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv7")(current_layer) 

    # Convolution 8
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv8")(current_layer) 

    # Convolution 9
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_regularizer=regularizers.l2(l2), name="Conv9")(current_layer) 
    shape = current_layer.get_shape().as_list()
    n_fc_nodes = shape[1] * shape[2]

    # Flattening
    current_layer = Flatten()(current_layer)

    # Concatenate STREAM_MAX
    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        current_layer = concatenate([current_layer, streamMaxInput])
        n_fc_nodes += 1

    # Fully connected 1
    current_layer = Dense(n_fc_nodes, activation=relu, name="Dense1")(current_layer)

    # Output layer
    current_layer = Dense(outputSize, name="Dense2")(current_layer)
    
    if problemType == "regression":
        o = Activation('linear', name='output_layer')(current_layer)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(current_layer)

    model = Model(inputs=inputs, outputs=o)
    return model

################################################
#   ConvNetQuakeINGV - Adapted for Speech8khz  #
################################################
def convnetquakeingv_speech8khz(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    l2 = 1e-3
    
    # Convolution 1
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv1")(waveformInput) 

    # Convolution 2
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv2")(current_layer) 

    # Convolution 3
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv3")(current_layer) 

    # Convolution 4
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv4")(current_layer) 

    # Convolution 5
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv5")(current_layer) 

    # Convolution 6
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv6")(current_layer) 

    # Convolution 7
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv7")(current_layer) 

    # Convolution 8
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv8")(current_layer) 

    # Convolution 9
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv9")(current_layer)  
    shape = current_layer.get_shape().as_list()
    n_fc_nodes = shape[1] * shape[2]

    # Flattening
    current_layer = Flatten()(current_layer)

    # Concatenate STREAM_MAX
    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        current_layer = concatenate([current_layer, streamMaxInput])
        n_fc_nodes += 1

    # Fully connected 1
    current_layer = Dense(n_fc_nodes, activation=relu, name="Dense1")(current_layer)

    # Output layer
    current_layer = Dense(outputSize, name="Dense2")(current_layer)
    
    if problemType == "regression":
        o = Activation('linear', name='output_layer')(current_layer)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(current_layer)

    model = Model(inputs=inputs, outputs=o)
    return model
    
#################################################
#   ConvNetQuakeINGV - Adapted for EMG dataset  #
#################################################
def convnetquakeingv_emg(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    
    # Convolution 1
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv1")(waveformInput) 

    # Convolution 2
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv2")(current_layer) 

    # Convolution 3
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv3")(current_layer) 

    # Convolution 4
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv4")(current_layer) 

    # Convolution 5
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv5")(current_layer) 

    # Convolution 6
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv6")(current_layer) 

    # Convolution 7
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv7")(current_layer) 

    # Convolution 8
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv8")(current_layer) 

    # Convolution 9
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="glorot_normal", kernel_regularizer=None, name="Conv9")(current_layer)  
    shape = current_layer.get_shape().as_list()
    n_fc_nodes = shape[1] * shape[2]

    # Flattening
    current_layer = Flatten()(current_layer)

    # Concatenate STREAM_MAX
    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        current_layer = concatenate([current_layer, streamMaxInput])
        n_fc_nodes += 1

    # Fully connected 1
    current_layer = Dense(n_fc_nodes, activation=relu, name="Dense1")(current_layer)

    # Output layer
    current_layer = Dense(outputSize, name="Dense2")(current_layer)
    
    if problemType == "regression":
        o = Activation('linear', name='output_layer')(current_layer)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(current_layer)

    model = Model(inputs=inputs, outputs=o)
    return model
    
###################################################
#   ConvNetQuakeINGV - Adapted for SP500 dataset  #
###################################################
def convnetquakeingv_sp500(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    l2 = 0.7
    
    # Convolution 1
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv1")(waveformInput) 

    # Convolution 2
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv2")(current_layer) 

    # Convolution 3
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv3")(current_layer) 

    # Convolution 4
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv4")(current_layer) 

    # Convolution 5
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv5")(current_layer) 

    # Convolution 6
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv6")(current_layer) 

    # Convolution 7
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv7")(current_layer) 

    # Convolution 8
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv8")(current_layer) 

    # Convolution 9
    current_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=True, activation=relu, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2), name="Conv9")(current_layer)  
    shape = current_layer.get_shape().as_list()
    n_fc_nodes = shape[1] * shape[2]

    # Flattening
    current_layer = Flatten()(current_layer)

    # Concatenate STREAM_MAX
    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        current_layer = concatenate([current_layer, streamMaxInput])
        n_fc_nodes += 1

    # Fully connected 1
    current_layer = Dense(n_fc_nodes, activation=relu, name="Dense1")(current_layer)

    # Output layer
    current_layer = Dense(outputSize, name="Dense2")(current_layer)
    
    if problemType == "regression":
        o = Activation('linear', name='output_layer')(current_layer)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(current_layer)

    model = Model(inputs=inputs, outputs=o)
    return model

##############
#   MagNet   #
##############
def magnet(inputShapes, outputSize=1, problemType="regression", useMaxStream=False, useCuDNNLSTM=False):    
    inputs = []
    waveformInput = Input(shape=inputShapes[0], name='waveform_input') 
    inputs.append(waveformInput)
    
    e = Conv1D(64, 3, padding = 'same', name="conv1d")(waveformInput) 
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)
     
    e = Conv1D(32, 3, padding = 'same', name="conv1d_1")(e) 
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)

    # Determine which LSTM implementation to use.
    if useCuDNNLSTM:
        from tensorflow.keras.layers import CuDNNLSTM
        e = Bidirectional(CuDNNLSTM(100, return_sequences=False))(e)
    else:
        from tensorflow.keras.layers import LSTM
        e = Bidirectional(LSTM(100, return_sequences=False))(e)
    

    if useMaxStream:
        streamMaxInput = Input(shape=inputShapes[1], name="stream_max_input")
        inputs.append(streamMaxInput)
        e = concatenate([e, streamMaxInput])
    e = Dense(outputSize)(e)

    if problemType == "regression":
        o = Activation('linear', name='output_layer')(e)
    elif problemType == "classification":
        o = Activation('softmax', name='output_layer')(e)
    
    model = Model(inputs=inputs, outputs=o)
    return model
