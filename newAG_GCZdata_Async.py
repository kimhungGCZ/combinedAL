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

import detection_engine as engine

import warnings
import time
import trollius as asyncio

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


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
start = time.time()

dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_dta_tsing.csv'
dataPath_result_relativeE = './results/relativeEntropy/realKnownCause/relativeEntropy_dta_tsing.csv'
dataPath_result_numenta = './results/numenta/realKnownCause/numenta_dta_tsing.csv'
dataPath_result_knncad = './results/knncad/realKnownCause/knncad_dta_tsing.csv'
dataPath_result_WindowGaussian = './results/windowedGaussian/realKnownCause/windowedGaussian_dta_tsing.csv'
dataPath_result_contextOSE = './results/contextOSE/realKnownCause/contextOSE_dta_tsing.csv'
dataPath_result_skyline = './results/skyline/realKnownCause/skyline_dta_tsing.csv'
dataPath_result_ODIN = './results/ODIN_result.csv'
# dataPath_result = './results/skyline/realKnownCause/skyline_dta_tsing.csv'
dataPath_raw = './data/realKnownCause/dta_tsing.csv'

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
sec_der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)

median_sec_der = np.median(sec_der)
std_sec_der = np.std(sec_der)

breakpoint_candidates = list(map(
    lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
    enumerate(sec_der)))
breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (
    np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

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
                            ["Bayes Algorithm", "Skyline Algorithm"],["Bayes Algorithm", "Numenta Algorithm"],
                            ["Bayes Algorithm", "ContextOSE Algorithm"],
                            ["Relative Entropy Algorithm", "Skyline Algorithm"],["Relative Entropy Algorithm", "Numenta Algorithm"],
                            ["Relative Entropy Algorithm", "ContextOSE Algorithm"]])
score_matrix = []
name_coff_metrix = []
for index, i in enumerate(raw_matrix):
    for in_cof_matrix in cof_matrix:
        score_matrix.append(i[0] * in_cof_matrix[0] + i[1] * in_cof_matrix[1])
        name_coff_metrix.append(
            [[raw_name_matrix[index][0], raw_name_matrix[index][1]], [in_cof_matrix[0], in_cof_matrix[1]]])

from multiprocessing.pool import ThreadPool

tasks = []
final_f = []
final_combination = []

for index, value in enumerate(score_matrix):
    new_data = result_dta_numenta.copy()
    new_data.anomaly_score = value
    #engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[index]);
    tasks.append(asyncio.Task(engine.anomaly_detection_asysn(new_data, raw_dta, name_coff_metrix[index])))
    #final_f.append(return_val)
    final_combination.append(name_coff_metrix[index])

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
end = time.time()
print("Total time: {}".format(end - start))


# print final_f
# final_index = np.argsort(final_f)[-1]
# filed_name = final_combination[final_index]
#
# print "%%%%%%%%%%%%%%%%%%%%%%%%______BEST CHOICE______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# print "Metric: %f * %s + %f * %s " % (filed_name[1][0], filed_name[0][0], filed_name[1][1], filed_name[0][1])
#
# new_data = result_dta_numenta.copy()
# new_data.anomaly_score = score_matrix[final_index]
# engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[final_index], 1)




# do some other stuff in the main process
