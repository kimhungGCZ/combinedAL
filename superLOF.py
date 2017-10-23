import numpy as np
import pandas as pd
import math as math
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb
from sets import Set
from sklearn.neighbors import DistanceMetric
import threading
import time
import os
import json

import detection_engine as engine
import scripts.obtain_data as data_engine

import warnings

warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


data = data_engine.getGCZDataFrame("2004DF").value.values
raw_data = data
data = np.array(data, dtype=np.float64)
data = np.reshape(data,(-1,1))
# raw_data = data

# Add X to real data
# data = np.reshape(raw_data, (-1,1))
#data = (data-min(data))/(max(data)-min(data))

# # fit the model
clf = LocalOutlierFactor(n_neighbors=1000)
y_pred = clf.fit_predict(data)
#y_pred_outliers = y_pred[200:]

#plt.plot(data)
#raw_y_pred[1142] = raw_y_pred[1143]
#plt.plot(data)
#raw_y_pred = (raw_y_pred-min(raw_y_pred))/(max(raw_y_pred)-min(raw_y_pred))
#y_pred = (y_pred-min(y_pred))/(max(y_pred)-min(y_pred))
#plt.scatter(np.arange(len(raw_y_pred)),raw_y_pred,c='red')
#plt.scatter(np.arange(len(y_pred)),y_pred,c='red')

# detected_outliners = (sorted(range(len(raw_y_pred)), key=lambda i: raw_y_pred[i])[:len(data_error)])
# correct_percentage = FF.correct_percentate((detected_outliners),np.concatenate(data_error))
# print("The error percentage: ", correct_percentage , "%")


plt.subplot(2, 1, 1)
plt.plot(raw_data,'b.-')
plt.title('Original Data')

plt.subplot(2, 1, 2)
plt.plot(y_pred, 'c')
plt.title('LoF Weighted Data')

plt.show()