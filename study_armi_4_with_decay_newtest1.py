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

import detection_engine as engine

import warnings

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


#DATA_FILE = 'dta_tsing'
DATA_FILE = 'data_1B3B8D'

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

# score_matrix = np.multiply(raw_matrix,cof_matrix)
# result_dta.anomaly_score = 0.3 * result_dta_contextOSE.anomaly_score + 0.7 * breakpoint_candidates + 0.0 * result_dta_bayes.anomaly_score
#
# threadLock = threading.Lock()
# threads = []
#
# for index, value in enumerate(score_matrix):
#    new_data = result_dta_numenta.copy()
#    new_data.anomaly_score = value
#    #engine.anomaly_detection(result_dta, raw_dta)
#    thread1 = myThread(new_data, raw_dta, name_coff_metrix[index])
#    thread1.start()
#    threads.append(thread1)
#
# for t in threads:
#     t.join()
# print "Exiting Main Thread"
from multiprocessing.pool import ThreadPool

pool = ThreadPool(processes=4)
final_f = []
final_combination = []
print "################## SOLELY SCORING STATE-OF-THE-ART ALGORITHMS ################################"
engine.calculate_point(DATA_FILE)
print "################## BUILDING THE METRIC WITH DEFAULT DECAY = 0.05 ################################"
################## BUILDING THE METRIC ################################
for index, value in enumerate(score_matrix):
    start_main_al = time.time()
    new_data = result_dta_numenta.copy()
    new_data.anomaly_score = value
    # engine.anomaly_detection(result_dta, raw_dta)
    async_result = pool.apply_async(engine.anomaly_detection,
                                    (new_data, raw_dta, name_coff_metrix[index], 0.05, DATA_FILE, 0))  # tuple of args for foo
    return_val = async_result.get()
    final_f.append(return_val)
    final_combination.append(name_coff_metrix[index])
    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al));
    print("_________________________________________________________________________________________")

final_index = np.argsort(final_f)[-1]
filed_name = final_combination[final_index]

print "%%%%%%%%%%%%%%%%%%%%%%%%______BEST CHOICE______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print "Metric %d: %f * %s + %f * %s " % (final_index, filed_name[1][0], filed_name[0][0], filed_name[1][1], filed_name[0][1])

############### To debug specific combination:############################
#final_index = 6
alpha = 0.05
print("Decay Value: %f" % alpha)
new_data = result_dta_numenta.copy()
new_data.anomaly_score = score_matrix[final_index]
start_main_al = time.time()
engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[final_index], alpha, DATA_FILE, 1)
end_main_al = time.time()
print("Execution time: {}".format(end_main_al - start_main_al));
print("_________________________________________________________________________________________")
print "%%%%%%%%%%%%%%%%%%%%%%%%______TESTING THE DECAY VALUE______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
####### TEST THE DECAY VALUE
alpha = 0.01
while alpha < 0.5:
    print("Decay Value: %f" %alpha)
    new_data = result_dta_numenta.copy()
    new_data.anomaly_score = score_matrix[final_index]
    start_main_al = time.time()
    engine.anomaly_detection(new_data, raw_dta, name_coff_metrix[final_index], alpha, DATA_FILE, 0)
    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al));
    print("_________________________________________________________________________________________")
    if alpha < 0.1:
        alpha = alpha + 0.01
    else:
        alpha = alpha + 0.05


end = time.time()
print("Total time: {}".format(end - start))

# do some other stuff in the main process
