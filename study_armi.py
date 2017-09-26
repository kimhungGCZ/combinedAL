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

dataPath = './data/realKnownCause/new_data_moisture.csv'

dataPath_result = './results/numentaTM/realKnownCause/numentaTM_dta_noiss.csv'
rawData1 = getCSVData(dataPath) if dataPath else None
result_dta = getCSVData(dataPath_result)if dataPath_result else None

# plt.plot(result_dta.anomaly_score[250:280])
# plt.plot(result_dta.anomaly_score)
# plt.show()


rawData = rawData1[:800]
rawData.value = cmfunc.realData();
rawData.to_csv('./data/realKnownCause/new_data_moisture_1.csv', index=False)
size = 500
anomaly_range = np.arange(250,280)

dta_full = rawData.iloc[:size]
dta_miss = dta_full.copy()
dta_noiss = dta_full.copy()
# anomalyData = np.asarray(map(lambda x: x if x!= 60 else None, anomalyData))
# anomalyData = pd.Series(anomalyData)
dta_miss.value[anomaly_range] = np.nan
dta_noiss.value[anomaly_range] = np.random.randint(30,61,len(anomaly_range)) + np.random.normal(0, 0.1, len(anomaly_range))
# Write to csv
#dta_noiss.to_csv('./data/realKnownCause/dta_noiss.csv', index=False)

# Add index
dta_miss.value.index = rawData.timestamp[:size]
dta_full.value.index = rawData.timestamp[:size]

mod_dtaMiss = sm.tsa.statespace.SARIMAX(dta_miss.value, order=(1,0,1),seasonal_order= (1, 1, 1, 12),enforce_stationarity=False,
                                                enforce_invertibility=False, dynamic=True)
res_dtaMiss = mod_dtaMiss.fit(disp=False)
print(res_dtaMiss.summary())
# Get prediction on all dataset
predict_dtaMiss = res_dtaMiss.get_prediction()
predicted_confident_dtaMiss = predict_dtaMiss.conf_int()
predicted_mean = predict_dtaMiss.predicted_mean
predicted_confident_precise_dtaMiss = np.array(map(lambda x: abs(x[1]-x[0]), predicted_confident_dtaMiss.values))


plt.figure(1)

# Plot data points
plt.subplot(211)
plt.plot(dta_miss.value.values)
plt.plot(predicted_mean.values,'g--')
plt.show()

prediction_error_origin = sum(abs(dta_full.value.values[anomaly_range] - predicted_mean.values[anomaly_range]))
MSE_score_origin = abs(dta_full.value.values - predicted_mean.values)
print("Reparing Error: ", prediction_error_origin, " Percentage Error: ", sum(MSE_score_origin)/sum(dta_full.value.values)*100)

anomaly_detected = np.array(result_dta.loc[result_dta['anomaly_score'] == 1].value.index)
anomaly_index = anomaly_detected if len(anomaly_detected) != 0 else np.array([np.argmax(result_dta.anomaly_score)])

print("Anomaly Point Found", anomaly_index)

anomaly_neighboor = np.array(cmfunc.kdd_Neigbors_2(dta_noiss,anomaly_index), dtype=np.int32)

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
