# -*- coding: utf-8 -*-
"""
Main

@author: Stefano Salati
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#import matplotlib
#%matplotlib inline
from itertools import compress
import traceback
from scipy.signal import lfilter, filtfilt
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import altair as alt

from load import *
from model import *
from model_epi import *


"""
Mains
"""
def main1():
    plt.close('all')
    
    # Caricamento dati
    
    if 0:
        keys = ['Italia',\
                'Spain', \
                'Germany', \
                'France', \
                'US', \
                'China'
                ]
    if 0:
        keys = ['Italia',\
                'EmiliaRomagna', \
                'Lombardia', \
                'Veneto', \
                #'Parma', \
                #'Modena', \
                ]
    if 1:
         keys = ['Italia']
         
         
    # Caricamento dati
    df_cases, df_deaths, df_recovered, X, Y_cases, Y_deaths, Y_recovered, Label_cases, Label_deaths, Label_recovered =  load(keys)
    prova = country(df_cases, df_deaths, df_recovered, "Italy")
    print(prova)
    altair_plot_for_confirmed_and_deaths(prova).interactive()
    
    
    #print(prova)
    #plt.plot(prova.index, prova['confirmed'])
    
    
    # Previsione casi
    days_ago = 0
    threshold = 50
    X_pred_cases, Y_pred_cases, Fit_cases, Rmse_cases, End_date_cases = run_time_model(X, Y_cases, past=days_ago, horizon=90, threshold=threshold)
    X_pred_deaths, Y_pred_deaths, Fit_deaths, Rmse_deaths, End_date_deaths = run_time_model(X, Y_deaths, past=days_ago, horizon=90, threshold=threshold)
    
    if 0:
        plot_timeseries("Casi", True, threshold,\
                        X, Y_cases, X_pred_cases, Y_pred_cases, Label_cases)
    
        print("Predicted end dates:")
        for i, date in enumerate(End_date_cases):
            print("{}\t\t(as of {},\t rmse={}):\t{}".format(Label_cases[i], daynumber2date(max(X[i])-days_ago), Rmse_cases[i], daynumber2date(date)))
    
    
    # Previsione morti
    if 0:
        plot_timeseries("Morti", True, threshold,\
                        X, Y_deaths, X_pred_deaths, Y_pred_deaths, Label_deaths)
        
        print("Predicted end dates (deaths):")
        for i, date in enumerate(End_date_deaths):
            print("{}\t\t(as of {},\t rmse={}):\t{}".format(Label_deaths[i], daynumber2date(max(X[i])-days_ago), Rmse_deaths[i], daynumber2date(date)))
            
    
    if 0:
        plot_timeseries2("Casi", True, threshold,\
                         X, Y_cases, X_pred_cases, Y_pred_cases, Label_cases,\
                         X, Y_deaths, X_pred_deaths, Y_pred_deaths, Label_deaths, scale2=7.7)
    
    
    # Come varia la previsione nel tempo
    if 0:
        run_time_model_timemachine(X, Y_cases, Label_cases, threshold=10)
    
    
    # Dati attuali
    N = 5
    b = N*[1.0/N]
    
    if 0:
        plot_crossseries("Crescita contagi vs contagi totali", "Cases", "New cases", \
                         np.log10([Y_cases[i][1:] for i in range(len(Y_cases))]), \
                         np.log10(lfilter(b, [1.], np.diff(Y_cases))), \
                         Label_cases)
    
    if 0:
        plot_crossseries("Crescita morti vs morti totali", "Cases", "New cases", \
                         np.log10([Y_deaths[i][1:] for i in range(len(Y_deaths))]), \
                         np.log10(lfilter(b, [1.], np.diff(Y_deaths))), \
                         Label_deaths)
    
    if 0:
        plot_crossseries("Morti vs casi totali", "Cases", "New cases", \
                         np.log10(Y_cases), \
                         np.log10(lfilter(b, [1.], Y_deaths)), \
                         Label_deaths)
    
    if 0:
        plot_crossseries("Crescita morti vs casi totali", "Cases", "New cases", \
                         np.log10([Y_cases[i][1:] for i in range(len(Y_cases))]), \
                         np.log10(lfilter(b, [1.], np.diff(Y_deaths))), \
                         Label_deaths)
            
    return


def main2():
    df_cases, df_deaths, df_recovered, X, Y_cases, Y_deaths, Y_recovered, Label_cases, Label_deaths, Label_recovered =  load([])
    df_pop = load_population()
    
    prova = country(df_cases, df_deaths, df_recovered, "Italy")
    print(prova)
    
    prova2 = trim_country(prova)
    print(prova2)
    
    #print(df_pop)
    #print(target_population(df_pop, 'Italy'))
    
    return



main2()
