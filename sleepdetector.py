import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #This is to force the model to use the CPU
import numpy as np
from models import sleepdetector_cnn_cpu, sleepdetector_lstm_cpu
import keras.backend as K
import tensorflow as tf
import scipy.stats




class Sleepdetector:
    
    def __init__(self, cnn_path = None, lstm_path = None, seq_length = 32):
        
        self.cnn = sleepdetector_cnn_cpu()
        if cnn_path is not None:
            self.cnn.load_weights(cnn_path)
        self.lstm = sleepdetector_lstm_cpu()
        if lstm_path is not None:
            self.lstm.load_weights(lstm_path)
            
        self.seq_length = seq_length
        self.iqr_target = [7.90, 11.37, 7.92, 11.56]
        self.med_target = [0.0257, 0.0942, 0.02157, 0.1055]
        
        self.get_cnn_output = K.function([self.cnn.layers[0].input, self.cnn.layers[1].input, self.cnn.layers[2].input, self.cnn.layers[3].input], [self.cnn.layers[53].output])
    
    
    
    def get_lstm_output(self, x_cnn):
        
        c = self.check_input_dimensions_lstm(x_cnn)
        
        if c == -1:
            return None
        
        n_examples = np.shape(x_cnn)[0]
        n_seqs = int(n_examples/self.seq_length)
        last_idx = n_seqs*self.seq_length
        x_lstm = np.reshape(x_cnn[0:last_idx], (-1, self.seq_length, 640))
        y_lstm = self.lstm.predict(x_lstm, verbose = 0)
        
        y_hat = np.argmax(np.reshape(y_lstm, (-1, 5)), axis = -1)
        
        return y_hat
        
        
        
    def check_input_dimensions_lstm(self, x_cnn):
       
        shape_x = np.shape(x_cnn)
        if len(shape_x) != 2:
            print("Input to LSTM must be of dimension 2")
            return -1
        
        if shape_x[0] <= 0:
            print("The first input must be a positive integer")
            return -1
        
        if shape_x[1] != 640:
            print("The second input must have a dimension = 640")
            return -1
        
        return 1
        
        
        
    
        
    def check_input_dimensions(self, x):
        #Check dimensions
        shape_x = np.shape(x)
        if len(shape_x) != 4:
            print("Input dimensions is different than 4")
            return -1
        
        if shape_x[0] != 4:
            print("First dimension should be equal to 4")
            return -1
        
        if shape_x[1] <= 0:
            print("Second dimension should be a positive integer")
            return -1
        
        if shape_x[2] != 3000:
            print("Third dimension should be equal to 3000")
            return -1
        
        if shape_x[3] != 1:
            print("Final dimension should be equal to 1")
            return -1
        
        return 1
    
    
    
        
        
    def predict(self, x):
        #Input x is of shape (4, n_examples, 3000, 1)
        
        #Check dimensions
        c = self.check_input_dimensions(x)
        
        if c == -1:
            print("Error in input dimensions")
            return -1
        
        #Scale Signal
        for i in range(4):
            x[i] = self.med_target[i] + (x[i] - np.median(x[i]))*(self.iqr_target[i]/scipy.stats.iqr(x[i]))
            
        
        x_cnn = self.get_cnn_output([x[0], x[1], x[2], x[3]])[0]
        y_lstm = self.get_lstm_output(x_cnn)
        
        if y_lstm is None:
            return -1
        
        return y_lstm
        
        
