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

import warnings

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


with open('data.json') as data_file:
    configure_data = json.load(data_file)
DATA_SET = configure_data[3]
DATA_FILE = str(DATA_SET['file_name']);
GROUND_TRUTH = list(DATA_SET['groud'])


# class myThread (threading.Thread):
#    def __init__(self, result_dta, raw_dta, file_name):
#       threading.Thread.__init__(self)
#       self.result_dta = result_dta
#       self.raw_dta = raw_dta
#       self.file_name = file_name
#    def run(self):
#       # Get lock to synchronize threads
#       threadLock.acquire()
#       engine.anomaly_detection(self.result_dta, self.raw_dta, self.file_name)
#       # Free lock to release next thread
#       threadLock.release()
if not os.path.exists('graph/' + DATA_FILE):
    os.makedirs('graph/' + DATA_FILE)
start = time.time()
dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_' + DATA_FILE + '.csv'
dataPath_result_relativeE = './results/relativeEntropy/realKnownCause/relativeEntropy_' + DATA_FILE + '.csv'
dataPath_result_numenta = './results/numenta/realKnownCause/numenta_' + DATA_FILE + '.csv'
dataPath_result_knncad = './results/knncad/realKnownCause/knncad_' + DATA_FILE + '.csv'
dataPath_result_WindowGaussian = './results/windowedGaussian/realKnownCause/windowedGaussian_' + DATA_FILE + '.csv'
dataPath_result_contextOSE = './results/contextOSE/realKnownCause/contextOSE_' + DATA_FILE + '.csv'
dataPath_result_skyline = './results/skyline/realKnownCause/skyline_' + DATA_FILE + '.csv'
dataPath_result_ODIN = './results/ODIN_result.csv'
# dataPath_result = './results/skyline/realKnownCause/skyline_'+DATA_FILE+'.csv'
dataPath_raw = './data/realKnownCause/' + DATA_FILE + '.csv'

result_dta_bayes = getCSVData(dataPath_result_bayes) if dataPath_result_bayes else None
result_dta_numenta = getCSVData(dataPath_result_numenta) if dataPath_result_numenta else None
result_dta_knncad = getCSVData(dataPath_result_knncad) if dataPath_result_knncad else None
result_dta_odin = getCSVData(dataPath_result_ODIN) if dataPath_result_ODIN else None
result_dta_relativeE = getCSVData(dataPath_result_relativeE) if dataPath_result_relativeE else None
result_dta_WindowGaussian = getCSVData(dataPath_result_WindowGaussian) if dataPath_result_WindowGaussian else None
result_dta_contextOSE = getCSVData(dataPath_result_contextOSE) if dataPath_result_contextOSE else None
result_dta_skyline = getCSVData(dataPath_result_skyline) if dataPath_result_skyline else None
raw_dta = getCSVData(dataPath_raw) if dataPath_raw else None

result_dta_numenta.anomaly_score[0:150] = np.min(result_dta_numenta.anomaly_score)

# dao ham bac 1
der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
# dao ham bac 2
sec_der = cmfunc.change_after_k_seconds(der, k=1)

median_sec_der = np.median(sec_der)
std_sec_der = np.std(sec_der)

breakpoint_candidates = list(map(
    lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
    enumerate(sec_der)))
breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (
    np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)
breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)
# result_dta = result_dta_numenta[150:400].copy()
result_dta = result_dta_numenta.copy()

# cof_matrix = np.array(np.matrix('1 0.75 0.5 0.25 0; 0 0.25 0.5 0.75 1'))
cof_matrix = np.array([[1, 0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0, 1]])

# raw_matrix = np.array([[breakpoint_candidates, result_dta_skyline.anomaly_score], [breakpoint_candidates, result_dta_numenta.anomaly_score], [breakpoint_candidates, result_dta_contextOSE.anomaly_score]])
raw_matrix = np.array([[breakpoint_candidates, result_dta_skyline.anomaly_score],
                       [breakpoint_candidates, result_dta_numenta.anomaly_score],
                       [breakpoint_candidates, result_dta_contextOSE.anomaly_score],
                       [result_dta_bayes.anomaly_score, result_dta_skyline.anomaly_score],
                       [result_dta_bayes.anomaly_score, result_dta_numenta.anomaly_score],
                       [result_dta_bayes.anomaly_score, result_dta_contextOSE.anomaly_score],
                       [result_dta_relativeE.anomaly_score, result_dta_skyline.anomaly_score],
                       [result_dta_relativeE.anomaly_score, result_dta_numenta.anomaly_score],
                       [result_dta_relativeE.anomaly_score, result_dta_contextOSE.anomaly_score]
                       ])
# raw_name_matrix = np.array([["EDGE Algorithm", "Skyline Algorithm"], ["EDGE Algorithm", "Numenta Algorithm"], ["EDGE Algorithm", "ContextOSE Algorithm"]])
raw_name_matrix = np.array([["EDGE Algorithm", "Skyline Algorithm"], ["EDGE Algorithm", "Numenta Algorithm"],
                            ["EDGE Algorithm", "ContextOSE Algorithm"],
                            ["Bayes Algorithm", "Skyline Algorithm"], ["Bayes Algorithm", "Numenta Algorithm"],
                            ["Bayes Algorithm", "ContextOSE Algorithm"],
                            ["Relative Entropy Algorithm", "Skyline Algorithm"],
                            ["Relative Entropy Algorithm", "Numenta Algorithm"],
                            ["Relative Entropy Algorithm", "ContextOSE Algorithm"]])
score_matrix = []
name_coff_metrix = []
for index, i in enumerate(raw_matrix):
    for in_cof_matrix in cof_matrix:
        score_matrix.append(i[0] * in_cof_matrix[0] + i[1] * in_cof_matrix[1])
        name_coff_metrix.append(
            [[raw_name_matrix[index][0], raw_name_matrix[index][1]], [in_cof_matrix[0], in_cof_matrix[1]]])

from multiprocessing.pool import ThreadPool

pool = ThreadPool(processes=4)
final_f = []
final_combination = []
print "################## SOLELY SCORING STATE-OF-THE-ART ALGORITHMS ################################"
engine.calculate_point(DATA_FILE, GROUND_TRUTH)
print "################## BUILDING THE METRIC WITH DEFAULT DECAY = 0.05 ################################"
################# BUILDING THE METRIC ################################
for index, value in enumerate(score_matrix):
    start_main_al = time.time()
    new_data = result_dta_numenta.copy()
    new_data.anomaly_score = value
    # engine.anomaly_detection(result_dta, raw_dta)
    async_result = pool.apply_async(engine.anomaly_detection,
                                    (new_data, raw_dta, name_coff_metrix[index], 0.05, GROUND_TRUTH, DATA_FILE, 0))  # tuple of args for foo
    return_val = async_result.get()
    final_f.append(return_val)
    final_combination.append(name_coff_metrix[index])
    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al));
    print("_________________________________________________________________________________________")

#final_f = [[100, 100, 100.0, 76, 100, 86.0], [100, 100, 100.0, 90, 100, 94.0], [50, 100, 66.0, 83, 50, 62.0], [50, 50, 50.0, 0, 0, 0], [50, 50, 50.0, 0, 0, 0], [100, 100, 100.0, 76, 100, 86.0], [100, 100, 100.0, 71, 100, 83.0], [40, 100, 57.0, 55, 50, 52.0], [40, 100, 57.0, 100, 50, 66.0], [25, 50, 33.0, 0, 0, 0], [100, 100, 100.0, 76, 100, 86.0], [100, 100, 100.0, 90, 100, 94.0], [12, 50, 19.0, 90, 100, 94.0], [9, 50, 15.0, 33, 20, 24.0], [0, 0, 0, 0, 0, 0], [33, 100, 49.0, 76, 100, 86.0], [40, 100, 57.0, 90, 100, 94.0], [100, 100, 100.0, 90, 100, 94.0], [33, 50, 39.0, 0, 0, 0], [50, 50, 50.0, 0, 0, 0], [33, 100, 49.0, 76, 100, 86.0], [33, 100, 49.0, 90, 100, 94.0], [28, 100, 43.0, 90, 100, 94.0], [33, 100, 49.0, 100, 10, 18.0], [25, 50, 33.0, 0, 0, 0], [33, 100, 49.0, 76, 100, 86.0], [28, 100, 43.0, 90, 100, 94.0], [15, 100, 26.0, 76, 100, 86.0], [11, 50, 18.0, 54, 60, 56.0], [0, 0, 0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [5, 50, 9.0, 0, 0, 0], [9, 50, 15.0, 0, 0, 0], [100, 50, 66.0, 0, 0, 0], [50, 50, 50.0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [5, 50, 9.0, 0, 0, 0], [14, 50, 21.0, 0, 0, 0], [25, 50, 33.0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [4, 50, 7.0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
final_index = np.argsort(final_f)[-1]

# df = pd.DataFrame(columns=('1', '2', '3','4','5','6','11', '22', '33','44','55','66','111', '222', '333','444','555','666','1111', '2222', '3333','4444','5555','6666','10', '20', '30','40','50','60'))
# tmp_array_1 = []
# for index,value in enumerate(final_f):
#     if (index == 0 or index % 5 != 0) and (index != len(final_f) -1):
#         tmp_array_1.extend(value)
#     elif index == len(final_f) -1:
#         tmp_array_1.extend(value)
#         df.loc[index / 5 + 1] = tmp_array_1
#     else:
#         df.loc[index/5] = tmp_array_1
#         tmp_array_1 = []
#         tmp_array_1.extend(value)
#
# df.to_csv('graph/' + DATA_FILE + '/resultLog.csv', index=False);

filed_name = final_combination[final_index]

print "%%%%%%%%%%%%%%%%%%%%%%%%______BEST CHOICE______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print "Metric %d: %f * %s + %f * %s " % (final_index, filed_name[1][0], filed_name[0][0], filed_name[1][1], filed_name[0][1])

############### To debug specific combination:############################
#final_index = 27
alpha = 0.05
print("Decay Value: %f" % alpha)
new_data = result_dta_numenta.copy()
new_data.anomaly_score = score_matrix[final_index]
start_main_al = time.time()
engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[final_index], alpha, GROUND_TRUTH, DATA_FILE , 1)
end_main_al = time.time()
print("Execution time: {}".format(end_main_al - start_main_al));
print("_________________________________________________________________________________________")
print "%%%%%%%%%%%%%%%%%%%%%%%%______TESTING THE DECAY VALUE______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
###### TEST THE DECAY VALUE
# alpha = 0.01
# while alpha < 0.5:
#     print("Decay Value: %f" %alpha)
#     new_data = result_dta_numenta.copy()
#     new_data.anomaly_score = score_matrix[final_index]
#     start_main_al = time.time()
#     engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[final_index], alpha, GROUND_TRUTH, DATA_FILE, 0)
#     end_main_al = time.time()
#     print("Execution time: {}".format(end_main_al - start_main_al));
#     print("_________________________________________________________________________________________")
#     if alpha < 0.1:
#         alpha = alpha + 0.01
#     else:
#         alpha = alpha + 0.05


end = time.time()
print("Total time: {}".format(end - start))

# do some other stuff in the main process
