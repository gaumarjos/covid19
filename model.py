# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:53:12 2020

Model

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


def daynumber2date(d):
    return str(dt.datetime(2019, 12, 31) + dt.timedelta(days=int(d)))[:10]

formatter_date = FuncFormatter(lambda x_val, tick_pos: "{}".format(daynumber2date(x_val)))
formatter_log10 = FuncFormatter(lambda x_val, tick_pos: "{}".format(np.power(10,x_val)))

def removenan(x, y):
    keep = np.logical_not(np.logical_or(np.logical_or(np.isnan(x), np.isinf(x)), np.logical_or(np.isnan(y), np.isinf(y))))
    x = list(compress(x, keep))
    y = list(compress(y, keep))
    if (sum(np.isnan(x)) + sum(np.isinf(x)) + sum(np.isnan(y) + sum(np.isinf(y)))) > 0:
        print("Stammerda non funziona")
    return x, y

def logistic1_model(x, a, dtau, tau):
    return a/(1+np.exp(-(x-dtau)/tau))

def logistic2_model(x, a, b, dtau, tau):
    return a/(1+b*np.exp(-np.power((x-dtau)/tau, 1.0)))

def logistic21_model(x, a, b, dtau, tau, ni):
    return a/np.power(1+b*np.exp(-np.power((x-dtau)/tau, 1.0)), 1.0/ni)

def logistic3_model(x, a, b, tau, alpha):
    # usato da Matteo P.
    return a/(1+b*np.exp(-np.power(x/tau, alpha)))

def logistic31_model(x, a, b, tau, alpha, ni):
    return a/np.power(1+b*np.exp(-np.power(x/tau, alpha)), 1.0/ni)

def logistic32_model(x, a, b, tau, alpha, ni, k):
    return k + (a-k)/np.power(1+b*np.exp(-np.power(x/tau, alpha)), 1.0/ni)

def linear_model(x, a, b):
    return a*x+b

def quadratic_model(x, a, b, c):
    return a*(x**2)+b*x+c

def model(model, x, y, horizon=0, threshold=10, relative_rmse = False, verbose=False):
    #x = np.asarray(x)
    #y = np.asarray(y)
    
    if len(x) > 0:
    
        if horizon == 0:
            x_pred = x
        else:
            x_pred = np.arange(min(x), max(x)+horizon, 1)
        
        if model == "logistic1":
            try:
                p0 = [20000, 100, 2]
                fit = curve_fit(logistic1_model, x, y, p0=p0)
                
                y_pred = logistic1_model(x_pred, fit[0][0], fit[0][1], fit[0][2])
    
                # End date
                sol = int(fsolve(lambda x : logistic1_model(x,fit[0][0],fit[0][1],fit[0][2]) - int(fit[0][2]),fit[0][1]))
                end_date = dt.datetime(2019, 12, 31) + dt.timedelta(days=sol)
                #print "End date: " + str(end_date)
                
            except:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[1, 0, 0], [0]]
                end_date = 0
        
        if model == "logistic2":
            try:
                p0 = [1.52646450e+05, 1.56215676e-01, 9.59401246e+01, 6.23161909e+00]
                fit = curve_fit(logistic2_model, x, y, maxfev=100000, p0=p0)
                
                y_pred = logistic2_model(x_pred, fit[0][0], fit[0][1], fit[0][2], fit[0][3])
                
            except Exception:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 1], [0]]
            
            end_date = 0
            
        if model == "logistic21":
            try:                
                #p0 = [1.52646450e+05, 1.56215676e-01, 9.59401246e+01, 6.23161909e+00, 1.0]
                # Uso il valore massimo dei casi come a0, questo fa si' che il modello converga per tutti i set, se no non convergerebbe
                p0 = [max(y), 1.56215676e-01, 9.59401246e+01, 6.23161909e+00, 1.0]
                bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0.000000],
                          [ np.inf,  np.inf,  np.inf,  np.inf, 8.000000])
                fit = curve_fit(logistic21_model, x, y, maxfev=100000, p0=p0)
                
                y_pred = logistic21_model(x_pred, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4])
                
            except Exception:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 1], [0]]
            
            end_date = 0
            
        if model == "logistic3":
            try:
                p0 = [1.52646560e+05, 1.56040732e-01, 9.59471514e+01, 1.5]
                bounds = ([-np.inf, -np.inf, -np.inf, 0.0000009],
                          [ np.inf,  np.inf,  np.inf, 8.0000000])
                fit = curve_fit(logistic3_model, x, y, maxfev=100000, bounds=bounds, p0=p0)
                
                y_pred = logistic3_model(x_pred, fit[0][0], fit[0][1], fit[0][2], fit[0][3])
                
            except Exception:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 1], [0]]
            
            end_date = 0
            
        if model == "logistic31":
            try:
                p0 = [1.52646560e+05, 1.56040732e-01, 9.59471514e+01, 1.5, 1.0]
                bounds = ([-np.inf, -np.inf, -np.inf, 0.0000009, 0.000000],
                          [ np.inf,  np.inf,  np.inf, 8.0000000, 8.000000])
                fit = curve_fit(logistic31_model, x, y, maxfev=100000, bounds=bounds, p0=p0)
                
                y_pred = logistic31_model(x_pred, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4])
                
            except Exception:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 1], [0]]
            
            end_date = 0
            
        if model == "logistic32":
            try:
                p0 = [1.52646560e+05, 1.56040732e-01, 9.59471514e+01, 1.5, 1.0, 0.0]
                bounds = ([-np.inf, -np.inf, -np.inf, 0.000009, 0.000000, -np.inf],
                          [ np.inf,  np.inf,  np.inf, 8.000000, 8.000000,  np.inf])
                fit = curve_fit(logistic32_model, x, y, maxfev=100000, bounds=bounds, p0=p0)
                
                y_pred = logistic32_model(x_pred, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5])
                
            except Exception:
                traceback.print_exc()
                print(model)
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 1], [0]]
            
            end_date = 0
        
        elif model == "linear":
            try:
                x_pred = np.arange(min(x), max(x)+horizon, 1)
                fit = curve_fit(linear_model, x, y)
                y_pred = linear_model(x_pred, fit[0][0], fit[0][1])
            except:
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0], [0]]
                
            end_date = 0
                
        elif model == "quadratic":
            try:
                x_pred = np.arange(min(x), max(x)+horizon, 1)
                fit = curve_fit(quadratic_model, x, y)
                y_pred = quadratic_model(x_pred, fit[0][0], fit[0][1], fit[0][2])
            except:
                x_pred = [0]
                y_pred = [0]
                fit = [[0, 0, 0], [0]]
                
            end_date = 0
        
        # Calcolo RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred[:len(y)]))
        if relative_rmse:
            rmse = rmse / max(y)
        
        # Cerco quando la derivata e' al di sotto di una certa soglia, quindi nuovi casi inferiori a ...
        d = np.diff(y_pred)
        zero_crossings = np.where(np.diff(np.sign(d-threshold)))[0]
        if len(zero_crossings)>0:
            last_zero_crossing_day = x_pred[zero_crossings[-1]]
            end_date = last_zero_crossing_day
        else:
            end_date = -1
            
        if verbose:
            #print len(x)
            #print len(y)
            #print len(x_pred)
            #print len(y_pred)
            print(fit[0])
        
        return x_pred, y_pred, fit, rmse, end_date
    
    else:
        return [], [], [], 0, 0


def run_time_model(X, Y, past=0, horizon=0, threshold=10, relative_rmse=False, verbose=False):
    Fit = list()
    X_pred = list()
    Y_pred = list()
    Rmse = list()
    End_date = list()
    
    if past>0:
        print("Non hai capito un cazzo, past deve essere < 0")
    
    for i, (x, y) in enumerate(zip(X,Y)):
        x, y = removenan(x, y)
        x = x[:len(x)+past]
        y = y[:len(y)+past]
        
        #x_pred, y_pred, fit, end_date = model("logistic1", x, y, horizon)
        x_pred, y_pred, fit, rmse, end_date = model("logistic21", x, y, horizon=horizon, threshold=threshold, relative_rmse=relative_rmse, verbose=verbose)
        #x_pred, y_pred, fit, end_date = model("logistic3", x, y, horizon)
        #x_pred, y_pred, fit, end_date = model("logistic31", x, y, horizon, verbose=True)
        #x_pred, y_pred, fit, end_date = model("logistic32", x, y, horizon, verbose=True)
        
        Fit.append(fit)
        X_pred.append(x_pred)
        Y_pred.append(y_pred)
        Rmse.append(rmse)
        End_date.append(end_date)
        
    return X_pred, Y_pred, Fit, Rmse, End_date


def run_time_model_timemachine(X, Y, Label, threshold):
    timemachine = range(-20,0+1,1)
    Timemachine_rmse = list()
    Timemachine_enddate = list()
        
    for past in timemachine:
        X_pred_cases, Y_pred_cases, Fit_cases, Rmse_cases, End_date_cases = run_time_model(X, Y, past=past, horizon=90, threshold=threshold, relative_rmse=True, verbose=False)
        Timemachine_rmse.append(Rmse_cases)
        Timemachine_enddate.append(End_date_cases)
            
    # Riordina risultati nel formato solito (per paese)
    paese = 0
    lista_enddate = list()
    lista_rmse = list()
    for paese in range(0, len(X)):
        lista_enddate_paese = list()
        lista_rmse_paese = list()
        for giorno in range(0, len(Timemachine_enddate)):
            lista_enddate_paese.append(Timemachine_enddate[giorno][paese])
            lista_rmse_paese.append(Timemachine_rmse[giorno][paese])
        lista_enddate.append(lista_enddate_paese)
        lista_rmse.append(lista_rmse_paese)
    
    # Grafico
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.yaxis.set_major_formatter(formatter_date)
    
    for i in range(len(lista_enddate)):
        ax1.plot(timemachine, lista_enddate[i], label=Label[i])
    ax1.legend()
    ax1.set_title("Fine prevista")
    ax1.xaxis.set_label_text("Giorno nel passato")
    ax1.yaxis.set_label_text("Giorno")
    ax1.yaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='minor')
    
    ax2 = fig.add_subplot(212)
    for i in range(len(lista_enddate)):
        ax2.plot(timemachine, lista_rmse[i], label=Label[i])
    ax2.set_title("RMSE")
    ax2.xaxis.set_label_text("Giorno nel passato")
    ax2.yaxis.set_label_text("RMSE")
    ax2.yaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='minor')
    
    return


def run_cross_model(Y1,Y2):
    Fit = list()
    Y1_pred = list()
    Y2_pred = list()
    
    for i, (y1, y2) in enumerate(zip(Y1, Y2)):
        y1, y2 = removenan(y1, y2)
        y1_pred, y2_pred, fit = model("linear", y1, y2)
        Fit.append(fit)
        Y1_pred.append(y1_pred)
        Y2_pred.append(y2_pred)
        
    return Y1_pred, Y2_pred, Fit


def plot_timeseries(title, logplot, threshold, \
                    X, Y, X_pred, Y_pred, Label):
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rc('font', size=14)
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    for idx, (x, y, x_pred, y_pred, label) in enumerate(zip(X, Y, X_pred, Y_pred, Label)):
        if logplot:
            ax1.scatter(x, np.log10(y), marker='.', color=colors[idx], label=label)
            ax1.plot(x_pred, np.log10(y_pred), color=colors[idx], label=label)
        else:
            ax1.scatter(x, y, marker='.', color=colors[idx], label=label)
            ax1.plot(x_pred, y_pred, color=colors[idx], label=label)
    
    ax1.set_title(title)
    ax1.xaxis.set_label_text("Giorno")
    ax1.yaxis.set_label_text("Casi")
    ax1.legend()
    ax1.xaxis.set_major_formatter(formatter_date)
    if logplot:
        ax1.set_ylim(0, 6)
        ax1.yaxis.set_major_formatter(formatter_log10)
    else:
        ax1.set_ylim(0, 1000000)
    ax1.yaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='minor')
    
    # Nuovi casi previsti (derivata della previsione comulativa)
    ax2 = fig.add_subplot(222)
    ax2.set_title(title + " (derivata)")
    for idx, (x, y, x_pred, y_pred, label) in enumerate(zip(X, Y, X_pred, Y_pred, Label)):
        d = np.diff(y_pred)
        ax2.plot(x_pred[1:], np.log10(d), color=colors[idx], label=label)
        
    ax2.xaxis.set_label_text("Giorno")
    ax2.yaxis.set_label_text("Nuovi casi")
    ax2.set_ylim(0, 5)
    ax2.xaxis.set_major_formatter(formatter_date)
    ax2.yaxis.set_major_formatter(formatter_log10)
    ax2.set_yticks([np.log10(threshold)], minor=True)
    ax2.xaxis.grid(True, which='major')
    ax2.xaxis.grid(True, which='minor')
    ax2.yaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='minor')
    
    # Errori
    ax3 = fig.add_subplot(223)
    ax3.set_title(title + " (errori)")
    for idx, (x, y, x_pred, y_pred, label) in enumerate(zip(X, Y, X_pred, Y_pred, Label)):
        x, y = removenan(x, y)
        ax3.plot(x, y_pred[:len(y)]-y, color=colors[idx], label=label)
        
    ax3.set_ylim(-3000, 3000)
    ax3.yaxis.grid(True, which='major')
    ax3.yaxis.grid(True, which='minor')
    
    plt.show()
    return


def plot_timeseries2(title, logplot, threshold, \
                     X1, Y1, X1_pred, Y1_pred, Label1, \
                     X2, Y2, X2_pred, Y2_pred, Label2, scale2=1.0):
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rc('font', size=14)
    fig = plt.figure()
    
    ax1 = fig.add_subplot(111)
    for idx, (x1, y1, x1_pred, y1_pred, label1, x2, y2, x2_pred, y2_pred, label2) in enumerate(zip(X1, Y1, X1_pred, Y1_pred, Label1, X2, Y2, X2_pred, Y2_pred, Label2)):
        if logplot:
            ax1.scatter(x1, np.log10(y1), marker='.', color=colors[idx], label=label1)
            ax1.plot(x1_pred, np.log10(y1_pred), color=colors[idx], label=label1)
            ax1.scatter(x2, np.log10(y2*scale2), marker='+', color=colors[idx], label=label2)
            ax1.plot(x2_pred, np.log10(y2_pred*scale2), color=colors[idx], label=label2)
        else:
            ax1.scatter(x1, y1, marker='.', color=colors[idx], label=label1)
            ax1.plot(x1_pred, y1_pred, color=colors[idx], label=label1)
            ax1.scatter(x2, y2*scale2, marker='+', color=colors[idx], label=label2)
            ax1.plot(x2_pred, y2_pred*scale2, color=colors[idx], label=label2)
    
    ax1.set_title(title)
    ax1.xaxis.set_label_text("Giorno")
    ax1.yaxis.set_label_text("Casi")
    ax1.legend()
    #ax1.xaxis.set_major_formatter(formatter_date)
    if logplot:
        ax1.set_ylim(0, 6)
        ax1.yaxis.set_major_formatter(formatter_log10)
    else:
        ax1.set_ylim(0, 200000)
    ax1.yaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='minor')
    
    plt.show()
    return


def plot_crossseries(title, xlabel, ylabel,
                     Y1, Y2, Label, Y1_pred=[], Y2_pred=[], Fit=[]):
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rc('font', size=14)
    
    if len(Y1_pred) > 0:
        fit_active = True
    else:
        fit_active = False
    
    for idx, (y1, y2, label) in enumerate(zip(Y1, Y2, Label)):
        #ax.scatter(y1, y2, marker='.', color=colors[idx], label=label)
        ax.plot(y1, y2, marker='.', color=colors[idx], label=label)
        
        if fit_active:
            y1_pred = Y1_pred[idx]
            y2_pred = Y2_pred[idx]
            fit = Fit[idx]
            ax.plot(y1_pred, y2_pred, color=colors[idx], label=label)
            #print("%s : % 5.2f %% deaths/total cases" %(label, fit[0][0]*100))  
              
    ax.legend()
    ax.set_title(title)
    ax.xaxis.set_label_text(xlabel)
    ax.yaxis.set_label_text(ylabel)
    ax.xaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    plt.show()
    
    return