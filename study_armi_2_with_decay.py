import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb

import warnings
warnings.simplefilter('ignore')

def getCSVData(dataPath):
  try:
    data = pd.read_csv(dataPath)
  except IOError("Invalid path to data file."):
    return
  return data

dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_dta_tsing.csv'
#dataPath_result = './results/relativeEntropy/realKnownCause/relativeEntropy_dta_tsing.csv'
dataPath_result_numenta = './results/numenta/realKnownCause/numenta_dta_tsing.csv'
#dataPath_result = './results/skyline/realKnownCause/skyline_dta_tsing.csv'
dataPath_raw = './data/realKnownCause/dta_tsing.csv'

result_dta_bayes = getCSVData(dataPath_result_bayes)if dataPath_result_bayes else None
result_dta_numenta = getCSVData(dataPath_result_numenta)if dataPath_result_numenta else None
raw_dta = getCSVData(dataPath_raw)if dataPath_raw else None

result_dta_numenta.anomaly_score[0:150] = np.min(result_dta_numenta.anomaly_score)

# dao ham bac 1
der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
# dao ham bac 2
sec_der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)

median_sec_der = np.median(sec_der)
std_sec_der = np.std(sec_der)

breakpoint_candidates = list(map(lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0, enumerate(sec_der)))
breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates))/(np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

breakpoint_candidates = np.insert(breakpoint_candidates, 0,0)

#result_dta = result_dta_numenta[150:400].copy()
result_dta = result_dta_numenta.copy()
#result_dta.anomaly_score = result_dta_numenta[150:400].anomaly_score  + result_dta_bayes[150:400].anomaly_score #breakpoint_candidates # + 0.5 *
result_dta.anomaly_score = 0.3*result_dta_numenta.anomaly_score + 0.7*breakpoint_candidates #+ result_dta_bayes.anomaly_score #breakpoint_candidates # + 0.5 *

# plt.subplot(411)
# plt.plot(result_dta_bayes.anomaly_score)
# plt.subplot(412)
# plt.plot(result_dta_numenta.anomaly_score)
# plt.subplot(413)
# plt.plot(result_dta.anomaly_score)
# plt.subplot(414)
# plt.plot(raw_dta.value)
# plt.show()


dta_full = result_dta

# dta_noiss = dta_full.copy()
# dta_miss = dta_full.copy()
# dta_truth = raw_dta['truth'].values
#
# dta_noiss.value.index = result_dta.timestamp
# dta_miss.value.index = result_dta.timestamp
dta_full.value.index = result_dta.timestamp
#dta_reverse.value.index = dta_reverse.timestamp

five_percentage = int((0.02* len(result_dta['anomaly_score'])) )
#anomaly_detected = np.array(result_dta.loc[result_dta['anomaly_score'] > 1].value.index)
np.argsort(result_dta['anomaly_score'])
#correct_detected = np.array(result_dta.loc[result_dta['anomaly_score'] == 0].value.index)
anomaly_index = np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
normal_index = np.array(np.argsort(result_dta['anomaly_score']))[:int((0.2* len(result_dta['anomaly_score'])) )]
#anomaly_index = [15, 143, 1860, 1700]

print("Anomaly Point Found", anomaly_index)
print("Correct Point Found", normal_index)
alpha = 0.05
Y = np.zeros(len(result_dta['anomaly_score']))
Z = np.zeros(len(result_dta['anomaly_score']))
X = list(map(lambda x: [x,result_dta.values[x][1]], np.arange(len(result_dta.values))))
tree = nb.KDTree(X, leaf_size=20)

for anomaly_point in anomaly_index:
  anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point), dtype=np.int32)
  for NN_pair in anomaly_neighboor:
    Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]

for normal_point in normal_index:
  anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, normal_point), dtype=np.int32)
  for NN_pair in anomaly_neighboor:
    Z[NN_pair[1]] = Z[NN_pair[1]] + (1-result_dta['anomaly_score'][normal_point]) - NN_pair[0] * alpha if (1-result_dta['anomaly_score'][normal_point]) - NN_pair[0] * alpha > 0 else Z[NN_pair[1]]

plt.subplot(411)
plt.plot(result_dta.anomaly_score, label='Metric of Score')
plt.legend(loc='upper center')
plt.subplot(412)
plt.plot(Y, label='Spreading Anomaly Score')
plt.legend(loc='upper center')
plt.subplot(413)
plt.plot(Z, label='Spreading Normal Score')
plt.legend(loc='upper center')
plt.subplot(414)
plt.plot(raw_dta.value, label='Final Score')
plt.legend(loc='upper center')
plt.show()

result_dta.anomaly_score = result_dta.anomaly_score + Y - Z

final_score = map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score);
#final_score = (final_score - np.min(final_score))/(np.max(final_score) - np.min(final_score - np.min(final_score)))


plt.subplot(511)
plt.plot(raw_dta.value, label='Raw data')
plt.legend(loc='upper center')
plt.subplot(512)
plt.plot(result_dta_bayes.anomaly_score, label='Bayes Result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,2))
plt.subplot(513)
plt.plot(result_dta_numenta.anomaly_score, label='Numenta Result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,2))
# plt.subplot(514)
# plt.plot(breakpoint_candidates)
# plt.legend(loc='upper center')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,0,2))
plt.subplot(515)
plt.plot(final_score, label='Our result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,max(final_score)))

plt.show()


plt.subplot(511)
plt.plot(raw_dta.value[720:800], label='Raw data')
plt.legend(loc='upper center')
plt.subplot(512)
plt.plot(result_dta_bayes.anomaly_score[720:800], label='Bayes Result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,2))
plt.subplot(513)
plt.plot(result_dta_numenta.anomaly_score[720:800], label='Numenta Result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,2))
# plt.subplot(514)
# plt.plot(breakpoint_candidates[720:800])
# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,0,2))
plt.subplot(515)
plt.plot(final_score[720:800], label='Our result')
plt.legend(loc='upper center')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,max(final_score[720:800])))
plt.show()


