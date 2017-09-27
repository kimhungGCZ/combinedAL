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
import trollius
import warnings
from numpy import mean, absolute

warnings.simplefilter('ignore')

# groud_trust = [[350, 832],[732, 733, 734, 745, 736, 755, 762, 773, 774, 795]]
groud_trust = [[581],
               [435, 460, 471, 557, 558, 559, 560, 561, 562, 563, 564, 570, 571, 572, 573, 574, 1174, 1175, 1383, 1418,
                1423]]


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)


def anomaly_detection(result_dta, raw_dta, filed_name, alpha, data_file='dta_tsing', debug_mode=0):
    if debug_mode == 1:
        dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_' + data_file + '.csv'
        dataPath_result_relativeE = './results/relativeEntropy/realKnownCause/relativeEntropy_' + data_file + '.csv'
        dataPath_result_numenta = './results/numenta/realKnownCause/numenta_' + data_file + '.csv'
        dataPath_result_knncad = './results/knncad/realKnownCause/knncad_' + data_file + '.csv'
        dataPath_result_WindowGaussian = './results/windowedGaussian/realKnownCause/windowedGaussian_' + data_file + '.csv'
        dataPath_result_contextOSE = './results/contextOSE/realKnownCause/contextOSE_' + data_file + '.csv'
        dataPath_result_skyline = './results/skyline/realKnownCause/skyline_' + data_file + '.csv'
        dataPath_result_ODIN = './results/ODIN_result.csv'
        # dataPath_result = './results/skyline/realKnownCause/skyline_data_compare_1.csv'
        dataPath_raw = './data/realKnownCause/' + data_file + '.csv'

        result_dta_bayes = getCSVData(dataPath_result_bayes) if dataPath_result_bayes else None
        result_dta_numenta = getCSVData(dataPath_result_numenta) if dataPath_result_numenta else None
        result_dta_knncad = getCSVData(dataPath_result_knncad) if dataPath_result_knncad else None
        result_dta_odin = getCSVData(dataPath_result_ODIN) if dataPath_result_ODIN else None
        result_dta_relativeE = getCSVData(dataPath_result_relativeE) if dataPath_result_relativeE else None
        result_dta_WindowGaussian = getCSVData(
            dataPath_result_WindowGaussian) if dataPath_result_WindowGaussian else None
        result_dta_contextOSE = getCSVData(dataPath_result_contextOSE) if dataPath_result_contextOSE else None
        result_dta_skyline = getCSVData(dataPath_result_skyline) if dataPath_result_skyline else None
        raw_dta = getCSVData(dataPath_raw) if dataPath_raw else None

        # result_dta_numenta.anomaly_score[0:150] = np.min(result_dta_numenta.anomaly_score)

        # dao ham bac 1
        der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
        # dao ham bac 2
        sec_der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)

        median_sec_der = np.median(sec_der)
        std_sec_der = np.std(sec_der)

        breakpoint_candidates = list(map(
            lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(
                std_sec_der) > 0 else 0,
            enumerate(sec_der)))
        breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (
            np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

        breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

    dta_full = result_dta

    dta_full.value.index = result_dta.timestamp

    std_anomaly_set = np.std(result_dta['anomaly_score'])
    np.argsort(result_dta['anomaly_score'])

    # Get 5% anomaly point
    # anomaly_index = np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])
    MAD_score = mad(result_dta['anomaly_score'])
    # anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3*1.4826 * MAD_score])

    if debug_mode == 1:
        cmfunc.plot_data_all('EXAMINATE DATA SET',
                             [[range(0, len(raw_dta.value)), raw_dta.value],
                              [groud_trust[1], raw_dta.value[groud_trust[1]]],
                              [groud_trust[0], raw_dta.value[groud_trust[0]]]],
                             ['lines', 'markers', 'markers'],
                             ['Raw data', 'Labeled Anomaly Point', 'Labeled Change Point'])

        cmfunc.plot_data_all('Abnormal Choosing Result',
                             [[range(0, len(raw_dta.value)), raw_dta.value],
                              [anomaly_index, raw_dta.value[anomaly_index]]],
                             ['lines', 'markers'], ['Raw Data', 'Anomaly points'])
    # print("Anomaly Point Found", anomaly_index)
    # Decay value is 5%
    # alpha = 0.1
    limit_size = int(1 / alpha)
    # Y is the anomaly spreding and Z is the normal spreading.
    Y = np.zeros(len(result_dta['anomaly_score']))
    Z = np.zeros(len(result_dta['anomaly_score']))
    X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(len(result_dta.values))))
    # dt=DistanceMetric.get_metric('pyfunc',func=mydist)
    tree = nb.KDTree(X, leaf_size=20)
    # tree = nb.BallTree(X, leaf_size=20, metric=dt)

    # Calculate Y
    for anomaly_point in anomaly_index:
        anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                     dtype=np.int32)
        for NN_pair in anomaly_neighboor:
            Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]

    backup_draw = result_dta.copy()

    # Calculate final score
    result_dta.anomaly_score = result_dta.anomaly_score + Y

    # Find normal point
    # normal_index = np.array(np.argsort(result_dta['anomaly_score']))[:int((0.4 * len(result_dta['anomaly_score'])))]
    normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
                    value <= np.percentile(result_dta['anomaly_score'], 20)]

    normal_index = np.random.choice(normal_index, int(len(normal_index) * 0.5), replace=False)
    if (debug_mode == 1):
        cmfunc.plot_data_all('Normal Choosing Result',
                             [[range(0, len(raw_dta.value)), raw_dta.value],
                              [normal_index, raw_dta.value[normal_index]]],
                             ['lines', 'markers'], ['a', 'b'])

    # Calculate Z
    for normal_point in normal_index:
        nomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, normal_point, limit_size),
                                    dtype=np.int32)
        for NN_pair in nomaly_neighboor:
            Z[NN_pair[1]] = Z[NN_pair[1]] + (1 - result_dta['anomaly_score'][normal_point]) - NN_pair[0] * alpha if (1 -
                                                                                                                     result_dta[
                                                                                                                         'anomaly_score'][
                                                                                                                         normal_point]) - \
                                                                                                                    NN_pair[
                                                                                                                        0] * alpha > 0 else \
                Z[NN_pair[1]]

    result_dta.anomaly_score = result_dta.anomaly_score - Z

    final_score = map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score);
    final_score = (final_score - np.min(final_score)) / (
        np.max(final_score) - np.min(final_score))

    ### Draw final result
    #### Draw step result ####

    if debug_mode == 1:
        cmfunc.plot_data('Final Result',
                         [raw_dta.value, result_dta_bayes.anomaly_score, breakpoint_candidates,
                          result_dta_relativeE.anomaly_score, result_dta_skyline.anomaly_score,
                          result_dta_numenta.anomaly_score, result_dta_contextOSE.anomaly_score, final_score], [],
                         ('Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result',
                          'Skyline Gaussian Result',
                          'Numenta Result', 'ContextOSE Result', 'Our Result'),
                         ['Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result', 'Skyline Result',
                          'Numenta Result', 'ContextOSE Result', 'Our Result'])
        # cmfunc.plot_data('Zoomed Final Result',
        #                  [raw_dta.value[720:800], result_dta_bayes.anomaly_score[720:800],
        #                   breakpoint_candidates[720:800],
        #                   result_dta_relativeE.anomaly_score[720:800], result_dta_skyline.anomaly_score[720:800],
        #                   result_dta_numenta.anomaly_score[720:800], result_dta_contextOSE.anomaly_score[720:800],
        #                   final_score[720:800]], [],
        #                  ('Raw Data[720:800]', 'Bayes Result[720:800]', 'EDGE Result [720:800]',
        #                   'Relative Entropy Result [720:800]', 'Skyline Result [720:800]', 'Numenta Result[720:800]',
        #                   'ContextOSE Result [720:800]',
        #                   'Our Result[720:800]'),
        #                  ['Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result', 'Skyline Result',
        #                   'Numenta Result', 'ContextOSE Result', 'Our Result'])
        cmfunc.plot_data('Step Result', [raw_dta.value, backup_draw.anomaly_score, Y, Z, final_score], [],
                         (
                             'Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score',
                             'Final Score'),
                         ['Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score',
                          'Final Score'])

    ### Find potential anomaly point
    std_final_point = np.std(final_score)
    anomaly_set = [i for i, v in enumerate(final_score) if v > 3 * std_final_point]

    # draw the whole data with potential anomaly point.
    if debug_mode == 1:
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
                anomaly_group_set[index_value - sliding_index] = np.append(
                    anomaly_group_set[index_value - sliding_index],
                    list(set(tmp_array).difference(anomaly_group_set[
                                                       index_value - sliding_index])))
                sliding_index = sliding_index + 1
            else:
                anomaly_group_set.append(np.sort(tmp_array))
        else:
            anomaly_group_set.append(np.sort(tmp_array))

    new_array = [tuple(row) for row in anomaly_group_set]
    uniques = new_array
    std_example_data = []
    std_example_outer = []
    detect_final_result = [[], []]
    for detect_pattern in uniques:
        # rest_anomaly_set = [i for i in anomaly_set if i not in list(detect_pattern)]
        list_of_anomaly = [int(j) for i in anomaly_group_set for j in i]
        example_data = [i for i in (
            list(raw_dta.value.values[list(z for z in range(int(min(detect_pattern) - 3), int(min(detect_pattern))) if z not in list_of_anomaly)]) + list(
                raw_dta.value.values[list(z for z in range(int(max(detect_pattern) + 1), int(max(detect_pattern) + 4)) if z not in list_of_anomaly)]))]

        in_std_with_Anomaly = np.std(
            example_data + list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern) + 1)]))
        std_example_data.append(in_std_with_Anomaly)

        example_data_iner = list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern)) + 1])

        in_std_with_NonAnomaly = np.std(example_data)
        if (in_std_with_Anomaly > 1.5 * in_std_with_NonAnomaly):
            detect_final_result[1].extend(np.array(detect_pattern, dtype=np.int))
        else:
            detect_final_result[0].append(int(np.min(detect_pattern)))
        std_example_outer.append(in_std_with_NonAnomaly)
    final_changepoint_set = detect_final_result[0]

    result_precision = 100 * len(set(final_changepoint_set).intersection(set(groud_trust[0]))) / len(
        set(final_changepoint_set)) if len(set(final_changepoint_set)) != 0 else 0
    result_recall = 100 * len(set(final_changepoint_set).intersection(set(groud_trust[0]))) / len(set(groud_trust[0]))
    result_f = float(2 * result_precision * result_recall / (result_precision + result_recall)) if (
                                                                                                       result_precision + result_recall) != 0 else 0
    ####################################################################################################################
    result_precision_AL = 100 * len(set(detect_final_result[1]).intersection(set(groud_trust[1]))) / len(
        set(detect_final_result[1])) if len(set(detect_final_result[1])) != 0 else 0
    result_recall_AL = 100 * len(set(detect_final_result[1]).intersection(set(groud_trust[1]))) / len(
        set(groud_trust[1]))
    result_f_AL = float(2 * result_precision_AL * result_recall_AL / (result_precision_AL + result_recall_AL)) if (
                                                                                                                      result_precision_AL + result_recall_AL) != 0 else 0
    ##################################################################################################
    print "Metric: %f * %s + %f * %s " % (filed_name[1][0], filed_name[0][0], filed_name[1][1], filed_name[0][1])
    print "Change Point Detection - Precision: %d %%, Recall: %d %%, F: %f" % (
        result_precision, result_recall, result_f)
    print "Anomaly Detection - Precision: %d %%, Recall: %d %%, F: %f" % (
        result_precision_AL, result_recall_AL, result_f_AL)
    print "Total Point: %f" % (np.mean([result_f, result_f_AL]))

    Grouping_Anomaly_Points_Result = [[range(0, len(raw_dta.value)), raw_dta.value]]
    Grouping_Anomaly_Points_Result_type = ['lines']
    bar_group_name = ['Raw Data']
    for j, value in enumerate(uniques):
        Grouping_Anomaly_Points_Result.append(
            list([list(map(int, value)), raw_dta.value.values[list(map(int, value))]]))
        Grouping_Anomaly_Points_Result_type.append('markers')
        bar_group_name.append("Group_" + str(j))

    # # Plot the grouping process.
    if debug_mode == 1:
        cmfunc.plot_data_all('Grouping Anomaly Points Result', Grouping_Anomaly_Points_Result,
                             Grouping_Anomaly_Points_Result_type, bar_group_name)

        # Plot the comparasion of std.
        cmfunc.plot_data_barchart("Anomaly Detection using Standard Deviation Changing",
                                  [[bar_group_name, std_example_data], [bar_group_name, std_example_outer]],
                                  name=['With potential anomaly', 'Non potential anomaly'])
    return np.mean([result_f, result_f_AL])

# def anomaly_detection_v2(result_dta, raw_dta, filed_name,data_file = 'dta_tsing', debug_mode = 0):
#
#     if debug_mode == 1:
#         dataPath_result_bayes = './results/bayesChangePt/realKnownCause/bayesChangePt_'+ data_file +'.csv'
#         dataPath_result_relativeE = './results/relativeEntropy/realKnownCause/relativeEntropy_'+ data_file +'.csv'
#         dataPath_result_numenta = './results/numenta/realKnownCause/numenta_'+ data_file +'.csv'
#         dataPath_result_knncad = './results/knncad/realKnownCause/knncad_'+ data_file +'.csv'
#         dataPath_result_WindowGaussian = './results/windowedGaussian/realKnownCause/windowedGaussian_'+ data_file +'.csv'
#         dataPath_result_contextOSE = './results/contextOSE/realKnownCause/contextOSE_'+ data_file +'.csv'
#         dataPath_result_skyline = './results/skyline/realKnownCause/skyline_'+ data_file +'.csv'
#         dataPath_result_ODIN = './results/ODIN_result.csv'
#         # dataPath_result = './results/skyline/realKnownCause/skyline_data_compare_1.csv'
#         dataPath_raw = './data/realKnownCause/'+ data_file +'.csv'
#
#         result_dta_bayes = getCSVData(dataPath_result_bayes) if dataPath_result_bayes else None
#         result_dta_numenta = getCSVData(dataPath_result_numenta) if dataPath_result_numenta else None
#         result_dta_knncad = getCSVData(dataPath_result_knncad) if dataPath_result_knncad else None
#         result_dta_odin = getCSVData(dataPath_result_ODIN) if dataPath_result_ODIN else None
#         result_dta_relativeE = getCSVData(dataPath_result_relativeE) if dataPath_result_relativeE else None
#         result_dta_WindowGaussian = getCSVData(
#             dataPath_result_WindowGaussian) if dataPath_result_WindowGaussian else None
#         result_dta_contextOSE = getCSVData(dataPath_result_contextOSE) if dataPath_result_contextOSE else None
#         result_dta_skyline = getCSVData(dataPath_result_skyline) if dataPath_result_skyline else None
#         raw_dta = getCSVData(dataPath_raw) if dataPath_raw else None
#
#         # result_dta_numenta.anomaly_score[0:150] = np.min(result_dta_numenta.anomaly_score)
#
#         # dao ham bac 1
#         der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
#         # dao ham bac 2
#         sec_der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
#
#         median_sec_der = np.median(sec_der)
#         std_sec_der = np.std(sec_der)
#
#         breakpoint_candidates = list(map(
#             lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(
#                 std_sec_der) > 0 else 0,
#             enumerate(sec_der)))
#         breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (
#             np.max(breakpoint_candidates) - np.min(breakpoint_candidates))
#
#         breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)
#
#     dta_full = result_dta
#
#     dta_full.value.index = result_dta.timestamp
#
#     std_anomaly_set = np.std(result_dta['anomaly_score'])
#     np.argsort(result_dta['anomaly_score'])
#
#     # Get 5% anomaly point
#     # anomaly_index = np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
#     anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])
#
#     #print("Anomaly Point Found", anomaly_index)
#     # Decay value is 5%
#     alpha = 0.1
#     limit_size = int(1 / alpha)
#     # Y is the anomaly spreding and Z is the normal spreading.
#     Y = np.zeros(len(result_dta['anomaly_score']))
#     Z = np.zeros(len(result_dta['anomaly_score']))
#     X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(len(result_dta.values))))
#     # dt=DistanceMetric.get_metric('pyfunc',func=mydist)
#     tree = nb.KDTree(X, leaf_size=20)
#     # tree = nb.BallTree(X, leaf_size=20, metric=dt)
#
#     # Calculate Y
#     for anomaly_point in anomaly_index:
#         anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
#                                      dtype=np.int32)
#         for NN_pair in anomaly_neighboor:
#             Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
#                 result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
#
#     backup_draw = result_dta.copy()
#
#
#
#     # Find normal point
#     # normal_index = np.array(np.argsort(result_dta['anomaly_score']))[:int((0.4 * len(result_dta['anomaly_score'])))]
#     normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
#                     value <= np.percentile(result_dta['anomaly_score'], 5)]
#
#     if (debug_mode == 1):
#         print("Correct Point Found", normal_index)
#         cmfunc.plot_data_all('Normal Choosing Result BEFORE',
#                              [[range(0, len(raw_dta.value)), raw_dta.value],
#                               [normal_index, raw_dta.value[normal_index]]],
#                              ['lines', 'markers'], ['a', 'b'])
#
#     normal_index = np.random.choice(normal_index, int(len(normal_index) * 0.5), replace=False)
#
#     if (debug_mode == 1):
#         cmfunc.plot_data_all('Normal Choosing Result AFTER',
#                              [[range(0, len(raw_dta.value)), raw_dta.value],
#                               [normal_index, raw_dta.value[normal_index]]],
#                              ['lines', 'markers'], ['a', 'b'])
#
#     # Calculate Z
#     for normal_point in normal_index:
#         nomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, normal_point, limit_size),
#                                     dtype=np.int32)
#         for NN_pair in nomaly_neighboor:
#             Z[NN_pair[1]] = Z[NN_pair[1]] + (1 - result_dta['anomaly_score'][normal_point]) - NN_pair[0] * alpha if (1 -
#                                                                                                                      result_dta[
#                                                                                                                          'anomaly_score'][
#                                                                                                                          normal_point]) - \
#                                                                                                                     NN_pair[
#                                                                                                                         0] * alpha > 0 else \
#                 Z[NN_pair[1]]
#             # Calculate final score
#
#     result_dta.anomaly_score = result_dta.anomaly_score + Y - Z
#
#     final_score = map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score);
#     final_score = (final_score - np.min(final_score)) / (
#     np.max(final_score) - np.min(final_score))
#
#     ### Draw final result
#     #### Draw step result ####
#
#     if debug_mode == 1:
#         cmfunc.plot_data('Final Result',
#                          [raw_dta.value, result_dta_bayes.anomaly_score, breakpoint_candidates,
#                           result_dta_relativeE.anomaly_score, result_dta_skyline.anomaly_score,
#                           result_dta_numenta.anomaly_score, result_dta_contextOSE.anomaly_score, final_score], [],
#                          ('Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result',
#                           'Skyline Gaussian Result',
#                           'Numenta Result', 'ContextOSE Result', 'Our Result'),
#                          ['Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result', 'Skyline Result',
#                           'Numenta Result', 'ContextOSE Result', 'Our Result'])
#         cmfunc.plot_data('Zoomed Final Result',
#                          [raw_dta.value[720:800], result_dta_bayes.anomaly_score[720:800],
#                           breakpoint_candidates[720:800],
#                           result_dta_relativeE.anomaly_score[720:800], result_dta_skyline.anomaly_score[720:800],
#                           result_dta_numenta.anomaly_score[720:800], result_dta_contextOSE.anomaly_score[720:800],
#                           final_score[720:800]], [],
#                          ('Raw Data[720:800]', 'Bayes Result[720:800]', 'EDGE Result [720:800]',
#                           'Relative Entropy Result [720:800]', 'Skyline Result [720:800]', 'Numenta Result[720:800]',
#                           'ContextOSE Result [720:800]',
#                           'Our Result[720:800]'),
#                          ['Raw Data', 'Bayes Result', 'EDGE Result', 'Relative Entropy Result', 'Skyline Result',
#                           'Numenta Result', 'ContextOSE Result', 'Our Result'])
#         cmfunc.plot_data('Step Result', [raw_dta.value, backup_draw.anomaly_score, Y, Z, final_score], [],
#                          (
#                              'Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score',
#                              'Final Score'),
#                          ['Raw Data', 'Metric of Score', 'Spreading Anomaly Score', 'Spreading Normal Score',
#                           'Final Score'])
#
#     ### Find potential anomaly point
#     std_final_point = np.std(final_score)
#     anomaly_set = [i for i, v in enumerate(final_score) if v > 3 * std_final_point]
#
#     # draw the whole data with potential anomaly point.
#     if debug_mode == 1:
#         cmfunc.plot_data_all('Potential Final Result',
#                              [[range(0, len(raw_dta.value)), raw_dta.value], [anomaly_set, raw_dta.value[anomaly_set]]],
#                              ['lines', 'markers'], ('Raw Data', 'High Potential Anomaly'))
#
#     # The algorithm to seperate anomaly point and change point.
#     X = list(map(lambda x: [x, x], np.arange(len(result_dta.values))))
#     newX = list(np.array(X)[anomaly_set])
#     newtree = nb.KDTree(X, leaf_size=20)
#
#     anomaly_group_set = []
#     new_small_x = 0
#     sliding_index = 1
#     for index_value, new_small_x in enumerate(anomaly_set):
#         anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point_1(newtree, X, new_small_x, anomaly_set),
#                                      dtype=np.int32)
#         tmp_array = list(map(lambda x: x[1], anomaly_neighboor))
#         if index_value > 0:
#             common_array = list(set(tmp_array).intersection(anomaly_group_set[index_value - sliding_index]))
#             # anomaly_group_set = np.concatenate((anomaly_group_set, tmp_array))
#             if len(common_array) != 0:
#                 union_array = list(set(tmp_array).union(anomaly_group_set[index_value - sliding_index]))
#                 anomaly_group_set[index_value - sliding_index] = np.append(
#                     anomaly_group_set[index_value - sliding_index],
#                     list(set(tmp_array).difference(anomaly_group_set[
#                                                        index_value - sliding_index])))
#                 sliding_index = sliding_index + 1
#             else:
#                 anomaly_group_set.append(np.sort(tmp_array))
#         else:
#             anomaly_group_set.append(np.sort(tmp_array))
#
#     new_array = [tuple(row) for row in anomaly_group_set]
#     uniques = new_array
#     std_example_data = []
#     std_example_outer = []
#     detect_final_result = [[],[]]
#     for detect_pattern in uniques:
#         #rest_anomaly_set = [i for i in anomaly_set if i not in list(detect_pattern)]
#         example_data = [i for i in (
#             list(raw_dta.value.values[int(min(detect_pattern) - 10): int(min(detect_pattern))]) + list(
#                 raw_dta.value.values[int(max(detect_pattern) + 1): int(max(detect_pattern) + 11)]))]
#         in_std_with_Anomaly = np.std(example_data + list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern) + 1)]))
#         std_example_data.append(in_std_with_Anomaly)
#         example_data_iner = list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern)) + 1])
#         example_data_outer = []
#         for j in example_data:
#             if j not in example_data_iner:
#                 example_data_outer.append(j)
#             else:
#                 example_data_iner.remove(j)
#
#         in_std_with_NonAnomaly = np.std(example_data_outer)
#         if (in_std_with_Anomaly > 2* in_std_with_NonAnomaly):
#             detect_final_result[1].extend(np.array(detect_pattern, dtype=np.int))
#         else:
#             detect_final_result[0].extend(np.array(detect_pattern, dtype=np.int))
#         std_example_outer.append(in_std_with_NonAnomaly)
#
#     #print("std with anomaly: ", std_example_data, " Std non anomaly", std_example_outer)
#     #print("Final result: ", detect_final_result)
#     result_precision = 100 * len(set(detect_final_result[0]).intersection(set(groud_trust[0]))) / len(set(detect_final_result[0])) if len(set(detect_final_result[0])) != 0 else 0
#     result_recall = 100 * len(set(detect_final_result[0]).intersection(set(groud_trust[0]))) / len(set(groud_trust[0]))
#     result_f  = float(2*result_precision*result_recall/(result_precision+result_recall)) if (result_precision+result_recall) != 0 else 0
#     ####################################################################################################################
#     result_precision_AL = 100 * len(set(detect_final_result[1]).intersection(set(groud_trust[1]))) / len(set(detect_final_result[1])) if len(set(detect_final_result[1])) != 0 else 0
#     result_recall_AL = 100 * len(set(detect_final_result[1]).intersection(set(groud_trust[1]))) / len(set(groud_trust[1]))
#     result_f_AL  = float(2*result_precision_AL*result_recall_AL/(result_precision_AL+result_recall_AL)) if (result_precision_AL+result_recall_AL) != 0 else 0
#     ##################################################################################################
#     print "Metric: %f * %s + %f * %s " %(filed_name[1][0], filed_name[0][0], filed_name[1][1], filed_name[0][1])
#     print "Change Point Detection - Precision: %d %%, Recall: %d %%, F: %f" %(result_precision, result_recall, result_f)
#     print "Anomaly Detection - Precision: %d %%, Recall: %d %%, F: %f" %(result_precision_AL, result_recall_AL, result_f_AL)
#     print "Total Point: %f" %(np.mean([result_f, result_f_AL]))
#     print("_________________________________________________________________________________________")
#     Grouping_Anomaly_Points_Result = [[range(0, len(raw_dta.value)), raw_dta.value]]
#     Grouping_Anomaly_Points_Result_type = ['lines']
#     bar_group_name = ['Raw Data']
#     for j, value in enumerate(uniques):
#         Grouping_Anomaly_Points_Result.append(
#             list([list(map(int, value)), raw_dta.value.values[list(map(int, value))]]))
#         Grouping_Anomaly_Points_Result_type.append('markers')
#         bar_group_name.append("Group_" + str(j))
#
#     # # Plot the grouping process.
#     if debug_mode == 1:
#         cmfunc.plot_data_all('Grouping Anomaly Points Result', Grouping_Anomaly_Points_Result,
#                              Grouping_Anomaly_Points_Result_type, bar_group_name)
#
#         # Plot the comparasion of std.
#         cmfunc.plot_data_barchart("Anomaly Detection using Standard Deviation Changing",
#                                   [[bar_group_name, std_example_data], [bar_group_name, std_example_outer]],
#                                   name=['With potential anomaly', 'Non potential anomaly'])
#     return np.mean([result_f, result_f_AL])
