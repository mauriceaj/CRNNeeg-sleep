import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import numpy as np
from sleepdetector import Sleepdetector


#Load Data File
filepath = './data/data.mat'
mat_file = sio.loadmat(filepath)
n_examples = np.shape(mat_file['sig1'])[0]
x = np.zeros((4, n_examples, 3000, 1))
x[0] = mat_file['sig1']
x[1] = mat_file['sig2']
x[2] = mat_file['sig3']
x[3] = mat_file['sig4']

#Load CRNNeeg
weights_cnn_path =  'cnn_weights.hdf5'
weights_lstm_path = 'lstm_weights.h5'
CRNNeeg = Sleepdetector(cnn_path = weights_cnn_path, lstm_path = weights_lstm_path)

#Predict and Plot Sleep Stages
y_hat = CRNNeeg.predict(x)
plt.plot(y_hat)

#Compare with original labels, compute Cohen's Kappa and Accuracy
labels_file = './data/labels.mat'
y_true = sio.loadmat(labels_file)['labels'].T
last_idx = np.shape(y_hat)[0]
n_max = np.minimum(last_idx, np.shape(y_true)[0])
y_true = y_true[:n_max] - 1

ck = cohen_kappa_score(y_true, y_hat)
accuracy = accuracy_score(y_true, y_hat)

print("Cohen's Kappa = %0.3f"%(ck))
print("Accuracy = %0.4f %%"%(100*accuracy))
