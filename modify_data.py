import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb
from sets import Set
import time
import datetime

import warnings

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data

dataPath_result= './data/realKnownCause/data_compare.csv'
result_dta = getCSVData(dataPath_result) if dataPath_result else None

result_dta = result_dta.drop('label', 1)
result_dta = result_dta.drop('truth', 1)
result_dta = result_dta.drop('isLabel', 1)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print st
time_array =  [ datetime.datetime.fromtimestamp(ts - 10000*i).strftime('%Y-%m-%d %H:%M:%S') for i in result_dta.timestamp.values]
result_dta['timestamp'] = time_array
print time_array
result_dta.to_csv("./data/realKnownCause/data_compare_1.csv", index=False);