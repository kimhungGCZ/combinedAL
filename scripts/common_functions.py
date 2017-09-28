import numpy as np
import sklearn.neighbors as nb
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
import math
from statsmodels.tsa.stattools import adfuller, acf, pacf
import requests, pandas as pd, numpy as np
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='kimhung1990', api_key='Of6D1v3klVr2tWI2piK8')

raw_data = [19.67860335,21.86011173,22.2450838,20.57687151,19.42195531,20.19189944,18.65201117,20.57687151,19.80692737,20.96184358,19.9352514,20.83351955,20.19189944,21.34681564,21.86011173,20.96184358,22.63005587,21.60346369,22.63005587,22.63005587,22.75837989,23.01502793,22.88670391,23.14335196,23.27167598,24.04162011,24.29826816,25.19653631,25.06821229,26.35145251,27.37804469,28.91793296,31.86938547,13.00575419,12.74910615,13.51905028,12.87743017,13.00575419,13.13407821,13.90402235,13.90402235,14.54564246,14.54564246,15.31558659,15.05893855,15.57223464,15.18726257,15.70055866,15.44391061,15.70055866,15.9572067,16.34217877,16.21385475,16.59882682,16.72715084,16.98379888,16.85547486,17.49709497,17.62541899,18.01039106,18.26703911,18.7803352,18.65201117,19.29363128,19.55027933,19.80692737,20.57687151,20.44854749,20.70519553,20.57687151,20.32022346,19.42195531,19.9352514,19.80692737,20.57687151,20.44854749,20.57687151,20.70519553,21.21849162,21.0901676,21.60346369,21.60346369,22.2450838,21.98843575,22.75837989,22.75837989,23.14335196,23.14335196,23.65664804,23.52832402,24.29826816,24.16994413,24.93988827,24.68324022,25.58150838,25.7098324,26.8647486,26.99307263,27.63469274,27.89134078,28.53296089,28.91793296,29.68787709,30.32949721,31.35608939,33.92256983,31.35608939,33.40927374,31.09944134,26.35145251,17.24044693,17.24044693,17.62541899,17.62541899,17.75374302,15.9572067,16.34217877,16.34217877,16.72715084,16.21385475,16.47050279,16.72715084,17.11212291,16.98379888,17.75374302,17.49709497,18.01039106,17.88206704,18.13871508,18.26703911,18.39536313,18.52368715,18.65201117,18.7803352,19.16530726,19.29363128,19.67860335,19.55027933,19.55027933,20.06357542,19.67860335,20.32022346,20.32022346,20.44854749,20.32022346,20.57687151,20.83351955,21.0901676,21.34681564,21.73178771,22.11675978,22.63005587,22.88670391,23.14335196,23.14335196,23.27167598,23.91329609,24.16994413,24.42659218,24.29826816,24.29826816,23.4,18.39536313,18.52368715,18.26703911,18.52368715,18.26703911,17.24044693,16.59882682,15.70055866,13.00575419,12.62078212,12.10748603,12.23581006,12.23581006,12.10748603,12.4924581,12.4924581,12.4924581,12.36413408,12.4924581,12.62078212,12.74910615,12.74910615,13.00575419,13.00575419,13.26240223,13.39072626,13.51905028,13.6473743,13.77569832,13.51905028,13.51905028,13.51905028,13.39072626,13.51905028,13.77569832,13.39072626,13.51905028,13.51905028,13.6473743,13.77569832,13.6473743,13.77569832,13.77569832,13.6473743,13.51905028,13.77569832,13.51905028,13.51905028,13.6473743,13.6473743,14.03234637,13.77569832,13.77569832,13.51905028,13.77569832,13.51905028,13.6473743,13.77569832,13.77569832,13.77569832,13.77569832,13.26240223,13.51905028,13.51905028,13.77569832,13.51905028,13.77569832,13.90402235,13.77569832,13.90402235,14.16067039,14.67396648,14.8022905,14.54564246,14.41731844,14.28899441,14.67396648,14.54564246,14.67396648,14.8022905,14.67396648,14.93061453,15.18726257,15.31558659,15.57223464,15.82888268,15.82888268,16.08553073,15.70055866,15.05893855,10.82424581,10.69592179,11.08089385,10.95256983,10.95256983,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,10.82424581,10.69592179,10.82424581,10.95256983,11.08089385,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,11.08089385,10.82424581,11.08089385,11.08089385,11.08089385,10.56759777,11.3375419,11.20921788,11.46586592,11.20921788,11.46586592,11.3375419,11.72251397,11.08089385,12.10748603,11.59418994,12.36413408,11.97916201,11.97916201,11.72251397,11.85083799,11.97916201,12.23581006,12.74910615,13.51905028,13.6473743,13.90402235,13.51905028,13.26240223,13.13407821,13.00575419,13.26240223,13.39072626,13.77569832,13.90402235,13.6473743,13.39072626,13.77569832,13.51905028,13.77569832,14.03234637,14.16067039,14.28899441,14.41731844,14.41731844,14.8022905,14.8022905,15.05893855,15.70055866,15.82888268,15.31558659,15.57223464,15.70055866,15.82888268,15.82888268,15.70055866,16.21385475,16.34217877,16.47050279,16.21385475,18.52368715,18.65201117,19.29363128,19.03698324,19.29363128,18.26703911,18.7803352,18.39536313,17.88206704,17.36877095,17.49709497,17.36877095,17.75374302,18.26703911,19.16530726,19.9352514,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,8.001117318,8.129441341,8.129441341,8.129441341,7.872793296,8.001117318,7.872793296,7.872793296,7.744469274,8.001117318,7.872793296,7.872793296,7.744469274,7.872793296,7.872793296,8.129441341,7.744469274,8.001117318,7.744469274,8.001117318,7.616145251,7.616145251,7.359497207,8.129441341,7.872793296,8.129441341,8.129441341,8.386089385,8.386089385,8.386089385,7.359497207,0.045027933,0.173351955,17.62541899,17.75374302,17.36877095,16.98379888,17.88206704,17.49709497,18.39536313,17.62541899,18.39536313,18.13871508,18.39536313,18.39536313,18.39536313,18.26703911,18.26703911,18.39536313,18.26703911,18.26703911,18.13871508,18.26703911,18.52368715,18.65201117,18.52368715,18.52368715,18.65201117,18.65201117,18.39536313,18.26703911,18.39536313,18.52368715,19.03698324,19.03698324,19.03698324,19.16530726,19.16530726,18.90865922,18.90865922,18.65201117,18.90865922,19.03698324,19.42195531,19.67860335,19.80692737,19.80692737,19.80692737,20.32022346,20.19189944,20.19189944,20.19189944,20.32022346,21.0901676,21.47513966,21.21849162,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.21849162,21.34681564,21.98843575,22.11675978,22.11675978,22.11675978,21.98843575,22.11675978,21.98843575,21.98843575,22.11675978,22.37340782,22.37340782,22.63005587,22.88670391,22.88670391,22.88670391,22.75837989,22.75837989,22.75837989,22.88670391,23.01502793,23.78497207,24.16994413,24.04162011,24.04162011,24.04162011,23.91329609,23.91329609,23.78497207,24.16994413,24.16994413,24.81156425,24.93988827,25.06821229,25.06821229,25.06821229,24.93988827,24.81156425,25.06821229,24.93988827,25.19653631,25.83815642,26.09480447,26.09480447,26.09480447,25.96648045,25.96648045,25.96648045,25.83815642,25.83815642,25.96648045,26.47977654,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,26.60810056,26.60810056,26.8647486,26.73642458,27.24972067,27.24972067,27.50636872,27.37804469,27.37804469,27.37804469,27.24972067,27.24972067,27.24972067,27.37804469,27.63469274,27.89134078,28.0196648,27.63469274,27.76301676,27.89134078,27.63469274,27.50636872,27.50636872,27.76301676,28.0196648,28.14798883,28.27631285,28.27631285,28.0196648,28.14798883,28.14798883,28.14798883,28.27631285,28.27631285,28.40463687,28.40463687,28.53296089,28.66128492,28.66128492,28.66128492,28.40463687,28.27631285,28.27631285,28.40463687,28.91793296,28.91793296,28.91793296,29.04625698,29.17458101,29.30290503,29.30290503,29.30290503,29.43122905,29.43122905,29.81620112,29.55955307,29.55955307,29.17458101,29.43122905,28.14798883,28.14798883,28.0196648,28.0196648,28.14798883,28.78960894,29.04625698,29.30290503,29.17458101,29.17458101,29.30290503,29.43122905,29.43122905,29.68787709,29.94452514,30.71446927,30.97111732,31.22776536,31.35608939,31.48441341,31.9977095,32.25435754,32.51100559,33.28094972,34.43586592,20.70519553,20.96184358,20.96184358,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.21849162,21.21849162,21.34681564,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.60346369,21.60346369,21.86011173,21.86011173,21.86011173,21.86011173,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.11675978,22.2450838,22.2450838,22.2450838,22.37340782,22.37340782,22.63005587,22.63005587,23.27167598,23.27167598,23.52832402,23.52832402,23.65664804,23.52832402,23.52832402,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.78497207,23.78497207,24.04162011,24.04162011,24.81156425,25.19653631,25.19653631,25.06821229,25.06821229,25.19653631,25.19653631,25.06821229,25.06821229,25.06821229,25.06821229,25.19653631,25.19653631,24.93988827,24.93988827,25.06821229,25.06821229,25.58150838,25.58150838,26.09480447,26.35145251,26.47977654,26.35145251,26.47977654,26.35145251,26.35145251,26.35145251,26.22312849,26.47977654,26.60810056,28.27631285,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,28.14798883,27.76301676,27.76301676,28.0196648,28.27631285,28.27631285,28.14798883,28.0196648,28.14798883,28.14798883,28.27631285,28.14798883,28.14798883,28.27631285,28.53296089,28.53296089,28.40463687,28.40463687,28.53296089,28.53296089,28.53296089,28.40463687,28.53296089,28.78960894,29.17458101,29.17458101,29.04625698,29.17458101,29.17458101,29.17458101,29.17458101,29.43122905,29.81620112,30.20117318,30.58614525,30.71446927,30.8427933,30.8427933,31.35608939,31.48441341,31.74106145,32.25435754,32.89597765,35.97575419,23.4,23.52832402,23.52832402,23.52832402,23.27167598,23.4,23.52832402,23.52832402,23.91329609,24.5549162,24.68324022,24.68324022,24.81156425,24.68324022,24.68324022,24.81156425,24.81156425,24.81156425,25.32486034,25.83815642,26.09480447,26.22312849,26.22312849,26.22312849,26.22312849,26.09480447,26.09480447,26.22312849,26.47977654,26.99307263,27.12139665,27.37804469,27.24972067,27.24972067,27.24972067,27.24972067,27.24972067,27.37804469,27.37804469,27.63469274,27.76301676,27.89134078,27.89134078,27.89134078,28.14798883,28.0196648,28.14798883,28.14798883,28.40463687,29.17458101,29.17458101,29.55955307,29.43122905,29.55955307,29.43122905,29.43122905,29.30290503,29.55955307,29.81620112,30.58614525,30.71446927,30.71446927,31.09944134,31.22776536,31.48441341,31.61273743,32.12603352,32.51100559,33.66592179,19.16530726,19.55027933,19.55027933,19.55027933,19.67860335,19.80692737,19.9352514,19.9352514,19.9352514,20.06357542,20.19189944,20.57687151,20.83351955,20.83351955,20.96184358,21.0901676,21.0901676,21.34681564,21.47513966,21.47513966,21.73178771,22.11675978,22.2450838,22.2450838,22.37340782,22.37340782,22.2450838,22.37340782,22.50173184,22.63005587,23.65664804,23.91329609,23.91329609,23.91329609,23.91329609,23.78497207,23.65664804,23.78497207,24.16994413,24.29826816,25.06821229,25.19653631,25.19653631,25.19653631,25.19653631,25.32486034,25.19653631,25.06821229,25.19653631,25.45318436,25.7098324,25.83815642,25.96648045,25.83815642,25.83815642,25.7098324,25.83815642,25.7098324,25.7098324,26.09480447,26.73642458,27.12139665,27.12139665,27.12139665,26.99307263,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,27.37804469,27.50636872,27.50636872,27.76301676,27.76301676,27.63469274,27.76301676,27.76301676,27.63469274,27.63469274,27.76301676,27.76301676,27.89134078,27.76301676,27.63469274,27.63469274,27.63469274,27.76301676,27.89134078,28.27631285,28.78960894,29.17458101,29.55955307,29.55955307,29.68787709,29.94452514,30.32949721,30.71446927,31.35608939,32.51100559,17.24044693,17.49709497,17.62541899,17.75374302,17.62541899,17.88206704,18.01039106,17.88206704,17.88206704,17.88206704,18.01039106,18.26703911,18.39536313,18.39536313,18.39536313,18.52368715,18.65201117,18.7803352,18.52368715,18.90865922,18.7803352,18.90865922,19.03698324,19.16530726,19.03698324,19.29363128,19.29363128,19.42195531,19.29363128,19.29363128,19.55027933,19.67860335,19.9352514,19.80692737,19.9352514,20.06357542,19.9352514,20.19189944,20.19189944,20.19189944,20.44854749,20.32022346,20.57687151,20.44854749,20.44854749,20.57687151,20.57687151,20.57687151,20.57687151,20.83351955,20.96184358,21.0901676,21.34681564,21.34681564,21.34681564,21.21849162,21.0901676,21.21849162,21.21849162,21.47513966,21.60346369,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.2450838,22.88670391,22.88670391,22.88670391,23.01502793,22.88670391,22.88670391,22.88670391,22.88670391,23.01502793,23.01502793,23.4,23.4,23.78497207,23.78497207,23.78497207,23.4,23.4,23.4,23.52832402,23.78497207,24.29826816,24.5549162,24.42659218,24.5549162,24.42659218,24.29826816,24.29826816,24.16994413,24.16994413,24.29826816,24.81156425,24.93988827,24.93988827,24.93988827,24.93988827,24.68324022,24.68324022,24.68324022,24.68324022,24.81156425,25.06821229,25.32486034,25.45318436,25.58150838,25.45318436,25.58150838,25.45318436,25.32486034,25.45318436,25.7098324,25.96648045,26.09480447,26.35145251,26.35145251,26.60810056,26.60810056,26.8647486,26.8647486,27.12139665,27.37804469,27.89134078,31.09944134,21.60346369,21.86011173,22.11675978,22.50173184,22.88670391,23.01502793,23.52832402,23.52832402,24.81156425,27.24972067,34.56418994,48.87527933,5.947932961]

def optimateParameter(data):
    print("-------------- Start Optimize Parameters ----------------")
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    aic_best = 999999
    param_best = 0
    seasonal_best = 0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit(disp=False)
                print('Optimize:  ', param ,'x',param_seasonal, " AIC: ", results.aic)
                if results.aic < aic_best:
                    aic_best = results.aic
                    param_best = param
                    seasonal_best = param_seasonal
            except:
                continue
    print('ARIMA{}x{} - AIC:{}'.format(param_best, seasonal_best, aic_best))
    return param_best, seasonal_best, aic_best

# This function is used to build and find the nearest neighbors of anomaly data.
def check_in_array(a, b):
    check_result = 0;
    for i in a:
        for j in b:
            if i == j[1]:
                check_result = 1
                break
    return check_result
def kdd_Neigbors(dta, index_ano):
    # the input data should be under Pandas dataframe format.
    #X = np.reshape(dta.value.values, (-1, 1))
    X = list(map(lambda x: [x, dta.values[x][1]], np.arange(len(dta.values))))
    tree = nb.KDTree(X, leaf_size=20)
    inverse_neighboor = []
    anomaly_index = index_ano
    for anomaly_index_in in anomaly_index:
        anomaly_point = X[anomaly_index_in]
        flag_stop = 0
        flag_round = 2
        while flag_stop <= 2:
            len_start = len(inverse_neighboor)
            dist, ind = tree.query([anomaly_point], k=flag_round)
            for i in ind[0]:
                if i not in inverse_neighboor:
                    in_dist, in_ind = tree.query([X[i]], k=flag_round)
                    #if ((anomaly_index_in in in_ind[0]) or (check_in_array(in_ind[0], inverse_neighboor)==1)):
                    if (anomaly_index_in in in_ind[0]):
                        inverse_neighboor = np.append(inverse_neighboor, i)
            flag_round += 1
            len_stop = len(inverse_neighboor)
            if len_start == len_stop:
                flag_stop += 1
            else:
                flag_stop = 0

    # dist, ind = tree.query(anomaly_point, k=30)
    #
    # print sorted(ind)  # indices of 3 closest neighbors
    # print dist  # distances to 3 closest neighbors
    print("Nearest Neighboor of Anomaly point ", len(inverse_neighboor))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(dta.value.values)
    plt.plot(np.array(inverse_neighboor, dtype=np.int32),dta.value.values[np.array(inverse_neighboor, dtype=np.int32)],'o')
    plt.plot(np.array(index_ano, dtype=np.int32), dta.value.values[np.array(index_ano, dtype=np.int32)],
             'x')
    plt.show()
    return inverse_neighboor

def kdd_Neigbors_2(dta, index_ano):
    # the input data should be under Pandas dataframe format.
    start_anp = index_ano;
    X = list(map(lambda x: [x,dta.values[x][1]], np.arange(len(dta.values))))
    #X = np.reshape(dta.value.values, (-1, 1))
    tree = nb.KDTree(X, leaf_size=20)
    flag_finding = 0
    initial_index = []
    while flag_finding == 0:
        new_anomaly_point = find_anomaly_point(tree, X, index_ano, initial_index)
        if (len(index_ano) < len(new_anomaly_point)):
            initial_index = index_ano
            index_ano = np.array(new_anomaly_point, dtype=np.int32)
        else:
            flag_finding = 1

    inverse_neighboor = index_ano
    print("Nearest Neighboor of Anomaly point ", len(inverse_neighboor))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(dta.value.values)
    plt.plot(np.array(inverse_neighboor, dtype=np.int32), dta.value.values[np.array(inverse_neighboor, dtype=np.int32)],
             'o')
    plt.plot(np.array(start_anp, dtype=np.int32), dta.value.values[np.array(start_anp, dtype=np.int32)],
             'x')
    plt.show()
    return inverse_neighboor


def find_anomaly_point(tree,X, index_ano,initial_index):
    inverse_neighboor = initial_index
    anomaly_index = index_ano
    for anomaly_index_in in anomaly_index:
        if anomaly_index_in not in initial_index:
            anomaly_point = X[anomaly_index_in]
            flag_stop = 0
            flag_round = 2
            while flag_stop <= 2:
                len_start = len(inverse_neighboor)
                dist, ind = tree.query([anomaly_point], k=flag_round)
                for i in ind[0]:
                    if i not in inverse_neighboor:
                        in_dist, in_ind = tree.query([X[i]], k=flag_round)
                        if ((anomaly_index_in in in_ind[0]) or (check_in_array(in_ind[0], inverse_neighboor) == 1)):
                            inverse_neighboor = np.append(inverse_neighboor, i)
                len_stop = len(inverse_neighboor)
                if len_start == len_stop:
                    flag_stop += 1
                    flag_round += 1
                else:
                    flag_stop = 0
                    flag_round = 2
    return inverse_neighboor

def find_inverneghboor_of_point(tree,X, index_ano, limit_size):
    inverse_neighboor = []
    anomaly_point = X[index_ano]
    flag_stop = 0
    flag_round = 2
    while flag_stop <= limit_size:
        len_start = len(inverse_neighboor)
        dist, ind = tree.query([anomaly_point], k=flag_round)
        for index_dist, i in enumerate(ind[0]):
            if [index_dist, i] not in inverse_neighboor:
                if inverse_neighboor != []:
                    if i not in [in_key[1] for in_key in inverse_neighboor]:
                        in_dist, in_ind = tree.query([X[i]], k=flag_round)
                        if ((index_ano in in_ind[0])) :#or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                            inverse_neighboor.append(
                                [index_dist, i])  # np.append(inverse_neighboor, [index_dist, i], axis=0)
                else:
                    in_dist, in_ind = tree.query([X[i]], k=flag_round)
                    if ((index_ano in in_ind[0])) :#or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                        inverse_neighboor.append(
                            [index_dist, i])  # np.append(inverse_neighboor, [index_dist, i], axis=0)
        len_stop = len(inverse_neighboor)
        if len_start == len_stop:
            flag_stop += 1
            flag_round += 1
        else:
            # Reset flag_stop and flag_round when found
            flag_stop = 0
        if len(inverse_neighboor) > limit_size:
            break

    return inverse_neighboor

def calcRMS(truthSeries, resultSeries, repairlist):
    cost = 0
    for i in range(0,len(truthSeries), 1):
        if i in repairlist:
            delta = resultSeries[i]- truthSeries[i]
            cost = cost + (delta*delta)

    cost = cost/ len(repairlist);

    return math.sqrt(cost);

def realData():
    return raw_data[0:800][::-1] #, [item for item in range(len(raw_data[0:800])) if raw_data[item] == 60]

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=52, center=False).mean()
    rolstd = timeseries.rolling(window=52, center=False).std()

    # # Plot rolling statistics:
    # orig = plt.plot(timeseries.values, color='blue', label='Original')
    # mean = plt.plot(rolmean.values, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd.values, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

def getCSVData(dataPath):
  try:
    data = pd.read_csv(dataPath)
  except IOError("Invalid path to data file."):
    return
  return data
def change_after_k_seconds(data, k=1):
    data1 = data[0:len(data) -k]
    data2 = data[k:]
    return list(map(lambda x: x[1] - x[0], zip(data1, data2)))

def plot_data(charName,data, mode, title = ('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5'), name = ['a', 'b', 'c', 'd', 'e', 'f']):
    fig = tools.make_subplots(rows=len(data), cols=1, subplot_titles=title)
    for index,in_data in enumerate(data):
        trace1 = go.Scatter(x=range(0,len(in_data)), y=in_data, name = name[index], mode  = 'lines' if mode == [] else mode[index])
        fig.append_trace(trace1, index+1, 1)

    fig['layout'].update(width=1200, title=charName)
    # Working online
    #py.plot(fig, filename=charName)
    # Working Offline
    plotly.offline.plot(fig, filename=charName)

def plot_data_all(charName,data, mode,name):
    fig = []
    for index,in_data in enumerate(data):
        trace1 = go.Scatter(x=in_data[0], y=in_data[1], name = name[index], mode  = 'lines' if mode[index] == None else mode[index], marker = dict(
        size = 15) if mode[index] == 'markers' else dict())
        fig.append(trace1)

    layout = dict(title=charName
                  )

    # Working online
    #py.plot(fig, filename=charName)
    # Working Offline
    plotly.offline.plot(dict(data=fig,layout=layout), filename=charName)

def plot_data_barchart(charName,data, name = ['a', 'b', 'c', 'd', 'e', 'f']):
    data_1 = []
    for index,in_data in enumerate(data):
        trace1 = go.Bar(x=in_data[0], y=in_data[1], name = name[index])
        data_1.append(trace1)
    layout = go.Layout(
        barmode='group',
        title=charName,
    )

    fig = go.Figure(data=data_1, layout=layout)
    # Working online
    #py.plot(fig, filename=charName)
    # Working Offline
    plotly.offline.plot(fig, filename=charName)

def find_inverneghboor_of_point_1(tree,X, index_ano, anomaly_set, limit_size):
    inverse_neighboor = []
    anomaly_point = X[index_ano]
    flag_stop = 0
    flag_round = 2
    while flag_stop <= 3:
        len_start = len(inverse_neighboor)
        dist, ind = tree.query([anomaly_point], k=flag_round)
        for index_dist, i in enumerate(ind[0]):
            if [index_dist, i] not in inverse_neighboor:
                if inverse_neighboor != []:
                    if i not in [in_key[1] for in_key in inverse_neighboor]:
                        in_dist, in_ind = tree.query([X[i]], k=flag_round)
                        if ((index_ano in in_ind[0]))or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                            if i in anomaly_set:
                                inverse_neighboor.append(
                                    [index_dist, i])
                                # np.append(inverse_neighboor, [index_dist, i], axis=0)
                else:
                    in_dist, in_ind = tree.query([X[i]], k=flag_round)
                    if ((index_ano in in_ind[0])) or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                        if i in anomaly_set:
                            inverse_neighboor.append(
                                [index_dist, i])  # np.append(inverse_neighboor, [index_dist, i], axis=0)
        len_stop = len(inverse_neighboor)
        if len_start == len_stop:
            flag_stop += 1
            flag_round += 1
        else:
            # Reset flag_stop and flag_round when found
            flag_stop = 0
            flag_round = 2

    return inverse_neighboor