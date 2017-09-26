import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb
from sets import Set

import warnings

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_dta_tsing.csv'
# dataPath_result = './results/relativeEntropy/realKnownCause/relativeEntropy_dta_tsing.csv'
dataPath_result_numenta = './results/numenta/realKnownCause/numenta_dta_tsing.csv'
# dataPath_result = './results/skyline/realKnownCause/skyline_dta_tsing.csv'
dataPath_raw = './data/realKnownCause/dta_tsing.csv'

result_dta_bayes = getCSVData(dataPath_result_bayes) if dataPath_result_bayes else None
result_dta_numenta = getCSVData(dataPath_result_numenta) if dataPath_result_numenta else None
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
# result_dta.anomaly_score = result_dta_numenta[150:400].anomaly_score  + result_dta_bayes[150:400].anomaly_score #breakpoint_candidates # + 0.5 *
result_dta.anomaly_score = 0.3 * result_dta_numenta.anomaly_score + 0.7 * breakpoint_candidates + 0 * result_dta_bayes.anomaly_score  # breakpoint_candidates # + 0.5 *

dta_full = result_dta

dta_full.value.index = result_dta.timestamp

five_percentage = int((0.02 * len(result_dta['anomaly_score'])))

np.argsort(result_dta['anomaly_score'])

anomaly_index = np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
normal_index = np.array(np.argsort(result_dta['anomaly_score']))[:int((0.2 * len(result_dta['anomaly_score'])))]

print("Anomaly Point Found", anomaly_index)
print("Correct Point Found", normal_index)
# Decay value is 5%
alpha = 0.05

# Y is the anomaly spreding and Z is the normal spreading.
Y = np.zeros(len(result_dta['anomaly_score']))
Z = np.zeros(len(result_dta['anomaly_score']))
X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(len(result_dta.values))))
tree = nb.KDTree(X, leaf_size=20)

# Calculate Y
for anomaly_point in anomaly_index:
    anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point), dtype=np.int32)
    for NN_pair in anomaly_neighboor:
        Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
            result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]

# Calculate Z
for normal_point in normal_index:
    anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, normal_point), dtype=np.int32)
    for NN_pair in anomaly_neighboor:
        Z[NN_pair[1]] = Z[NN_pair[1]] + (1 - result_dta['anomaly_score'][normal_point]) - NN_pair[0] * alpha if (1 -
                                                                                                                 result_dta[
                                                                                                                     'anomaly_score'][
                                                                                                                     normal_point]) - \
                                                                                                                NN_pair[
                                                                                                                    0] * alpha > 0 else \
            Z[NN_pair[1]]

backup_draw = result_dta.copy()

# Calculate final score
result_dta.anomaly_score = result_dta.anomaly_score + Y - Z

final_score = map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score);
final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score - np.min(final_score)))

### Draw final result
#### Draw step result ####
cmfunc.plot_data('Step Result', [raw_dta.value, backup_draw.anomaly_score, Y, Z, final_score], [],
                 ('Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score', 'Final Score'),
                 ['Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score', 'Final Score'])
cmfunc.plot_data('Final Result',
                 [raw_dta.value, breakpoint_candidates, result_dta_numenta.anomaly_score, final_score], [],
                 ('Raw Data', 'EDGE Result', 'Numenta Result', 'Final Score'),
                 ['Raw Data', 'EDGE Result', 'Numenta Result', 'Final Score'])
cmfunc.plot_data('Zoomed Final Result', [raw_dta.value[720:800], breakpoint_candidates[720:800],
                                         result_dta_numenta.anomaly_score[720:800], final_score[720:800]], [],
                 ('Raw Data[720:800]', 'EDGE Result[720:800]', 'Numenta Result[720:800]', 'Final Score[720:800]'),
                 ['Raw Data', 'Bayes Result', 'Numenta Result', 'Final Score'])

### Find potential anomaly point
std_final_point = np.std(final_score)
anomaly_set = [i for i, v in enumerate(final_score) if v > 3 * std_final_point]

# draw the whole data with potential anomaly point.
cmfunc.plot_data_all('Potential Final Result',
                     [[range(0, len(raw_dta.value)), raw_dta.value], [anomaly_set, raw_dta.value[anomaly_set]]],
                     ['lines', 'markers'], ('Raw Data', 'High Potential Anomaly'))

# The algorithm to seperate anomaly point and change point.
X = list(map(lambda x: [x, x], np.arange(len(result_dta.values))))
newX = list(np.array(X)[anomaly_set])
newtree = nb.KDTree(X, leaf_size=20)

anomaly_group_set = []
new_small_x = 0
sliding_index = 1
for index_value, new_small_x in enumerate(anomaly_set):
    anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point_1(newtree, X, new_small_x, anomaly_set),
                                 dtype=np.int32)
    tmp_array = list(map(lambda x: x[1], anomaly_neighboor))
    if index_value > 0:
        common_array = list(set(tmp_array).intersection(anomaly_group_set[index_value - sliding_index]))
        # anomaly_group_set = np.concatenate((anomaly_group_set, tmp_array))
        if len(common_array) != 0:
            union_array = list(set(tmp_array).union(anomaly_group_set[index_value - sliding_index]))
            anomaly_group_set[index_value - sliding_index] = np.append(anomaly_group_set[index_value - sliding_index],
                                                                       list(set(tmp_array).difference(anomaly_group_set[
                                                                                                          index_value - sliding_index])))
            sliding_index = sliding_index + 1
        else:
            anomaly_group_set.append(np.sort(tmp_array))
    else:
        anomaly_group_set.append(np.sort(tmp_array))

new_array = [tuple(row) for row in anomaly_group_set]
uniques = np.unique(new_array)
std_example_data = []
std_example_outer = []
for detect_pattern in uniques:
    rest_anomaly_set = [i for i in anomaly_set if i not in list(detect_pattern)]
    example_data = [i for i in (
        list(raw_dta.value.values[int(min(detect_pattern) - 10): int(min(detect_pattern))]) + list(
            raw_dta.value.values[int(max(detect_pattern) + 1): int(max(detect_pattern) + 11)])) if
                    i not in raw_dta.value.values[rest_anomaly_set]]
    std_example_data.append(
        np.std(example_data + list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern) + 1)])))
    example_data_iner = list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern)) + 1])
    example_data_outer = []
    for j in example_data:
        if j not in example_data_iner:
            example_data_outer.append(j)
        else:
            example_data_iner.remove(j)
    std_example_outer.append(np.std(example_data_outer))

print("std with anomaly: ", std_example_data, " Std non anomaly", std_example_outer)
Grouping_Anomaly_Points_Result = [[range(0, len(raw_dta.value)), raw_dta.value]]
Grouping_Anomaly_Points_Result_type = ['lines']
bar_group_name = ['Raw Data']
for j, value in enumerate(uniques):
    Grouping_Anomaly_Points_Result.append(list([list(map(int, value)), raw_dta.value.values[list(map(int, value))]]))
    Grouping_Anomaly_Points_Result_type.append('markers')
    bar_group_name.append("Group_" + str(j))

# Plot the comparasion of std.
cmfunc.plot_data_barchart("Anomaly Detection using Standard Deviation Changing",
                          [[bar_group_name, std_example_data], [bar_group_name, std_example_outer]],
                          name=['With potential anomaly', 'Non potential anomaly'])

# Plot the grouping process.
cmfunc.plot_data_all('Grouping Anomaly Points Result', Grouping_Anomaly_Points_Result,
                     Grouping_Anomaly_Points_Result_type, bar_group_name)
