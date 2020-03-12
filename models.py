from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Add, Activation
from keras.layers import TimeDistributed, Bidirectional, LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import keras



def sleepdetector_cnn_cpu(n_filters = [8, 16, 32], kernel_size = [50, 8, 8], Fs = 100, n_classes = 5):

    input_sig1 = Input(shape=(30*Fs,1))
    input_sig2 = Input(shape=(30*Fs,1))
    input_sig3 = Input(shape=(30*Fs,1))
    input_sig4 = Input(shape=(30*Fs,1))

    ## Each input will go through 1 CNNs
    
    ### First Input
    x0 = Conv1D(n_filters[0], 
              kernel_size = kernel_size[0], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(input_sig1)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    x0 = MaxPooling1D(pool_size=8, strides=None)(x0)
    
    x0 = Conv1D(n_filters[1], 
              kernel_size= kernel_size[1], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x0)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    x0 = MaxPooling1D(pool_size=8, strides=None)(x0)
    
    x0 = Conv1D(n_filters[2], 
              kernel_size= kernel_size[2], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x0)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    x0 = MaxPooling1D(pool_size=8, strides=None)(x0)
    
    ### Second Input
    x1 = Conv1D(n_filters[0], 
              kernel_size= kernel_size[0], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(input_sig2)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=8, strides=None)(x1)
    
    x1 = Conv1D(n_filters[1], 
              kernel_size= kernel_size[1], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=8, strides=None)(x1)
    
    x1 = Conv1D(n_filters[2], 
              kernel_size= kernel_size[2], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=8, strides=None)(x1) 
    
    ### Third Input
    x2 = Conv1D(n_filters[0], 
              kernel_size= kernel_size[0], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(input_sig3)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=8, strides=None)(x2)
    
    x2 = Conv1D(n_filters[1], 
              kernel_size= kernel_size[1], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=8, strides=None)(x2)
    
    x2 = Conv1D(n_filters[2], 
              kernel_size= kernel_size[2], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=8, strides=None)(x2)
    
    
    ### Fourth Input
    x3 = Conv1D(n_filters[0], 
              kernel_size= kernel_size[0], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(input_sig4)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling1D(pool_size=8, strides=None)(x3)
    
    x3 = Conv1D(n_filters[1], 
              kernel_size= kernel_size[1], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling1D(pool_size=8, strides=None)(x3)
    
    x3 = Conv1D(n_filters[2], 
              kernel_size= kernel_size[2], 
              strides = 1, 
              padding='same', 
              kernel_initializer='glorot_uniform')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling1D(pool_size=8, strides=None)(x3)

    merged_vector = keras.layers.concatenate([x0, x1, x2, x3], axis=-1)
    flattened_vector = Flatten()(merged_vector)
    final_x = Dense(n_classes, activation='softmax')(flattened_vector)
    
    with tf.device('/cpu:0'):
        model = Model(inputs=[input_sig1, input_sig2, input_sig3, input_sig4], outputs=[final_x])
    
    return model
    
    
def sleepdetector_lstm_cpu(timesteps = 32, vec_len = 640, n_units = 64, n_layers = 4):
    
    x_in = Input(shape = (timesteps, vec_len))
    
    x = Bidirectional(LSTM(units = n_units,
                return_sequences=True))(x_in)
    
    for i in range(n_layers - 1):
        x = Bidirectional(LSTM(units = n_units,
                return_sequences=True))(x)
    
    
    final_x = TimeDistributed(Dense(5, activation = 'softmax'))(x)   
    
    
    with tf.device('/cpu:0'):
        model = Model(inputs=[x_in], outputs=[final_x])
    
    return model
    
    
