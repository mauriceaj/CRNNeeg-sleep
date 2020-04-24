# CRNNeeg-sleep

## Introduction:
This page contains the necessary code to run CRNNeeg, a deep learning algorithm for automatically sleep staging PSG and long-term scalp EEG recordings. CRNNeeg is composed of a convolutional neural network (CNN) for feature extraction, followed by a recurrent neural network (RNN) that extracts the temporal dependencies of sleep stages.

CRNNeeg is implemented in Python (3.6), running Keras (2.2.3) with Tensorflow (1.12.0) as backend. Currently, only the cpu implementation of CRNNeeg is provided.

## How to use:
Step 1: Import the class Sleepdetector
```
from sleepdetector import Sleepdetector
```

Step 2: Construct an instance of Sleepdetector, and load its weights (weights are included in this repo)
```
weights_cnn_path =  'cnn_weights.hdf5'
weights_lstm_path = 'lstm_weights.h5'
CRNNeeg = Sleepdetector(cnn_path = weights_cnn_path, lstm_path = weights_lstm_path)
```

Step 3: Predict sleep stages using the method .predict() and input **x**
```
y_hat = CRNNeeg.predict(x)
```

**x** is a sequence of consecutive 30s EEG epochs that consists of 4 channels: F3-C3, C3-O1, F4-C4, and C4-O2.

**x** has a shape of (4, n, 3000, 1), where *4* corresponds to the number of channels, *n* corresponds to the number of 30s epochs, and *3000* corresponds to the number of samples of each 30s segment (= time x sampling_frequency = 30s x 100Hz = 3000)

The output **yhat** is an array of sleep stages, where 4 = Awake, 3 = REM, 2 = N1, 1 = N2, and 0 = N3.

An example is provided in **main.py**, where CRNNeeg is applied on the PSG recording 'abc-baseline-900001' of the ABC dataset[1][2][3].

## Environment:
Python 3.6

Tensorflow 1.12.0

Keras 2.2.3

## Contact:
Maurice Abou Jaoude

Department of Neurology

Massachusetts General Hospital

Email: maboujaoude(at)mgh.harvard.edu

## References:

1. Dean DA 2nd, Goldberger AL, Mueller R, Kim M, Rueschman M, Mobley D, Sahoo SS, Jayapandian CP, Cui L, Morrical MG, Surovec S, Zhang GQ, Redline S. Scaling Up Scientific Discovery in Sleep Medicine: The National Sleep Research Resource. Sleep. 2016 May 1;39(5):1151-64. doi: 10.5665/sleep.5774. Review. PubMed PMID: 27070134; PubMed Central PMCID: PMC4835314.

2. Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S. The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc. 2018 May 31. doi: 10.1093/jamia/ocy064. [Epub ahead of print] PubMed PMID: 29860441.

3. Bakker JP, Tavakkoli A, Rueschman M, Wang W, Andrews R, Malhotra A, Owens RL, Anand A, Dudley KA, Patel SR. Gastric Banding Surgery versus Continuous Positive Airway Pressure for Obstructive Sleep Apnea: A Randomized Controlled Trial. Am J Respir Crit Care Med. 2018 Apr 15;197(8):1080-1083. doi: 10.1164/rccm.201708-1637LE. PubMed PMID: 29035093; PubMed Central PMCID: PMC5909166.


