import urllib2, json
import numpy as np
import pandas as pd
import time
import datetime
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data

request = urllib2.Request("https://server.humm-box.com/api/devices/1B3B30/fastmeasures?fields=[content_volume]")
request.add_header("Authorization", "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTQ5MjcyODgyOTQ0NjU4MzExNiIsImF1ZCI6IkxMSWVDYXpIVEpTOG9kVW1kaHJHMmVuV3dQaW5yajUxIiwiZXhwIjoxNTEwMDI2NDU1LCJpYXQiOjE1MDY0MjY0NTV9.gma7GCb2-dYiMnJkeapyrd2Y_xhk0Wk_14zS49Yk7Pc")
result = urllib2.urlopen(request)
tem_data = json.load(result.fp)
tem_data.sort(key=lambda x: x[0])
data = [i[1] for index,i in enumerate(tem_data) if i[1]!= None and index not in range(581,601)]
plt.plot(data)
plt.show()
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
time_array =  [datetime.datetime.fromtimestamp(ts - 10000*i).strftime('%Y-%m-%d %H:%M:%S') for i in data]
d = {'timestamp': time_array, 'value': data}
df = pd.DataFrame(data=d)
df.to_csv("./data/realKnownCause/data_1B3B8D.csv", index=False);
result.close()