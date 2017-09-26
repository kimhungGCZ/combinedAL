import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc


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

result_dta = result_dta_numenta.copy()
result_dta.anomaly_score = result_dta_numenta.anomaly_score + result_dta_bayes.anomaly_score

#plt.plot(result_dta.anomaly_score[250:280])
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

five_percentage = int((0.01* len(result_dta['anomaly_score'])) )
#anomaly_detected = np.array(result_dta.loc[result_dta['anomaly_score'] > 1].value.index)
np.argsort(result_dta['anomaly_score'])
#correct_detected = np.array(result_dta.loc[result_dta['anomaly_score'] == 0].value.index)
anomaly_index = np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
normal_index = np.array(np.argsort(result_dta['anomaly_score']))[:five_percentage]
#anomaly_index = [15, 143, 1860, 1700]

print("Anomaly Point Found", anomaly_index)
print("Correct Point Found", normal_index)
anomaly_neighboor = np.array(cmfunc.kdd_Neigbors_2(result_dta,anomaly_index), dtype=np.int32)
print(anomaly_neighboor)
dta_miss.value[anomaly_neighboor] = np.nan
dta_reverse = dta_miss.loc[::-1]

# Add index
dta_noiss.value.index = result_dta.timestamp
dta_miss.value.index = result_dta.timestamp
dta_full.value.index = result_dta.timestamp
dta_reverse.value.index = dta_reverse.timestamp
"""
# Calculate the model with inverse data.
#param_best, seasonal_best, aic_best = cmfunc.optimateParameter(dta_reverse.value)
mod_dtaMiss_inverse = sm.tsa.statespace.SARIMAX(dta_reverse.value, order=(1,1,0),seasonal_order=(0,1,1,1))
res_dtaMiss_inverse = mod_dtaMiss_inverse.fit(disp=False)
print(res_dtaMiss_inverse.summary())
# Get prediction on all dataset
predict_dtaMiss_inverse = res_dtaMiss_inverse.get_prediction(dynamic=False)
predicted_confident_dtaMiss_inverse = predict_dtaMiss_inverse.conf_int()
predicted_mean_inverse = predict_dtaMiss_inverse.predicted_mean
predicted_confident_precise_dtaMiss_inverse = np.array(map(lambda x: abs(x[1]-x[0]), predicted_confident_dtaMiss_inverse.values))
predicted_mean_inverse = predicted_mean_inverse.loc[::-1]

# Calculate the model with miss data
#param_best, seasonal_best, aic_best = cmfunc.optimateParameter(dta_miss.value)
mod_dtaMiss = sm.tsa.statespace.SARIMAX(dta_miss.value, order=(1,1,1),seasonal_order=(0,1,0,1))
res_dtaMiss = mod_dtaMiss.fit(disp=False)
print(res_dtaMiss.summary())
# Get prediction on all dataset
predict_dtaMiss = res_dtaMiss.get_prediction(dynamic=False)
predicted_confident_dtaMiss = predict_dtaMiss.conf_int()
predicted_mean = predict_dtaMiss.predicted_mean
predicted_confident_precise_dtaMiss = np.array(map(lambda x: abs(x[1]-x[0]), predicted_confident_dtaMiss.values))


predicted_final_data = (predicted_mean.values + predicted_mean_inverse.values)/2;

RMS = cmfunc.calcRMS(dta_truth, predicted_final_data, anomaly_neighboor)

print("Final RMS: ", RMS)

plt.figure(1)

# Plot data points
plt.subplot(211)
plt.plot(predicted_final_data, label = "Repaired Value")
plt.plot(dta_truth, label = "Truth Value")
plt.legend()
#plt.plot(dta_reverse.value.values)
#plt.subplot(212)

#plt.plot(predicted_mean.values,'g--')
plt.show()

# prediction_error_origin = sum(abs(dta_full.value.values[anomaly_range] - predicted_mean.values[anomaly_range]))
# MSE_score_origin = abs(dta_full.value.values - predicted_mean.values)
# print("Reparing Error: ", prediction_error_origin, " Percentage Error: ", sum(MSE_score_origin)/sum(dta_full.value.values)*100)




# Add confident score to anomaly score based on confident range:
#anomaly_neighboor_score = np.column_stack((anomaly_neighboor,predicted_confident_precise_dtaMiss[np.array(anomaly_neighboor)]))
# Add confident score to anomaly score based on MSE Scoring:
anomaly_neighboor_score = map(lambda x: [x,MSE_score_origin[int(x)]], anomaly_neighboor)
#Sort array with confident score
anomaly_neighboor_score = sorted(anomaly_neighboor_score, key=lambda x: x[1])
for j in anomaly_neighboor_score:
    array_point_repairing = [np.int32(j[0])]
    for i in array_point_repairing:
        predict = res_dtaMiss.get_prediction(dta_full.timestamp[i],dta_full.timestamp[i])

        dta_miss.value[i] = predict.predicted_mean.values[0]
        dta_noiss.value[i]= predict.predicted_mean.values[0]
        result_dta.anomaly_score[i] = 0
        print("*****Predicted Value: ",predict.predicted_mean.values[0], " MSE: ", abs(dta_full.value.values[i] - predict.predicted_mean.values[0]))

    mod = sm.tsa.statespace.SARIMAX(dta_miss.value, order=(1,0,1),seasonal_order= (1, 1, 1, 12))
    res = mod.fit(disp=False)
    #print(res.summary())

    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    predicted_mean = predict.predicted_mean

    prediction_error = sum(abs(dta_full.value.values[anomaly_range] - predicted_mean.values[anomaly_range]))
    MSE_score = abs(dta_full.value.values - predicted_mean.values)
    print("Reparing Error Saved: ", prediction_error_origin - prediction_error, "Percentaged of MSE: Saved", (sum(MSE_score_origin) - sum(MSE_score))/sum(dta_full.value.values)*100)
    # plt.figure(1)
    #
    # # Plot data points
    # plt.subplot(211)
    # plt.plot(dta_miss.value.values)
    # # plt.plot(predicted_mean.values,'g--')
    # plt.subplot(212)
    # # plt.plot(Origin_data.value[:size])
    # plt.plot(dta_noiss.value.values)
    # plt.show()

"""

#####################################################################################################################
########################################################### ARIMA ###################################################
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from statsmodels.tsa.stattools import adfuller, acf, pacf
from datetime import date
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

ts_week = dta_full.value
ts_week.dropna(inplace=True)
ts_week_log = np.log(ts_week)
cmfunc.test_stationarity(ts_week_log)
ts_week_log_diff = ts_week_log - ts_week_log.shift()
ts_week_log_diff.dropna(inplace=True)
cmfunc.test_stationarity(ts_week_log_diff)

# # Plot the autocorrelation function (ACF) and partial autocorrelation function (PACF)
# lag_acf = acf(ts_week_log_diff, nlags=10)
# lag_pacf = pacf(ts_week_log_diff, nlags=10, method='ols')
#
#
# # Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
#
# # Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
# plt.show()


model = ARIMA(ts_week_log, order=(1, 1, 1))
results_ARIMA = model.fit(disp=-1)
print('RSS: %.4f'% sum((results_ARIMA.fittedvalues.values-ts_week_log_diff.values[:len(results_ARIMA.fittedvalues.values)])**2))


print(results_ARIMA.summary())
# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
#residuals.plot(kind='kde')
print(residuals.describe())

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_week_log.ix[0], index=ts_week_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_week.values)
plt.plot(predictions_ARIMA.values)
RMS = cmfunc.calcRMS(dta_truth, predictions_ARIMA.values, anomaly_neighboor)
print("RMS", RMS)
#print('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA.values-ts_week.values)**2)/len(ts_week.values)))
plt.show()