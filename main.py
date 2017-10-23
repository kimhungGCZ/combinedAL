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


#if not os.path.exists('graph/' + DATA_FILE):
#    os.makedirs('graph/' + DATA_FILE)
start = time.time()
raw_dataframe = data_engine.getGCZDataFrame("2004DF")
raw_dta = raw_dataframe.value.values

# dao ham bac 1
#der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
# dao ham bac 2
sec_der = cmfunc.change_after_k_seconds_with_abs(raw_dta, k=1)

median_sec_der = np.median(sec_der)
std_sec_der = np.std(sec_der)

breakpoint_candidates = list(map(
    lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
    enumerate(sec_der)))
breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (
    np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=4)
final_f = []
final_combination = []

############### To debug specific combination:############################
final_index = 0
alpha = 0.05
print("Decay Value: %f" % alpha)
new_data = raw_dataframe.copy()
new_data = new_data.assign(anomaly_score=pd.Series(breakpoint_candidates).values)
start_main_al = time.time()
engine.online_anomaly_detection(new_data, raw_dataframe, alpha)
end_main_al = time.time()
print("Execution time: {}".format(end_main_al - start_main_al));

end = time.time()
print("Total time: {}".format(end - start))

# do some other stuff in the main process
