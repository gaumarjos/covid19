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
Main
"""
def main():
    
    df_cases, df_deaths, df_recovered, X, Y_cases, Y_deaths, Y_recovered, Label_cases, Label_deaths, Label_recovered =  load([])
    df_pop = load_population()
    
    
    """
    Initial conditions
    """
    df_target_country = country(df_cases, df_deaths, df_recovered, "Italy")
    df_target_country = trim_country(df_target_country)
    target_population = get_target_population(df_pop, 'Italy')
    print(df_target_country)
    
    S0, E0, I0, R0, D0 = target_population, 5 * float(df_target_country.confirmed[0]), float(df_target_country.confirmed[0]), 0., 0.
    y0_sir = S0 / target_population, I0 / target_population, R0  # SIR IC array
    y0_sird = S0 / target_population, I0 / target_population, R0, D0  # SIRD IC array
    y0_seir = S0 / target_population, E0 / target_population, I0 / target_population, R0  # SEIR IC array
    y0_seird = S0 / target_population, E0 / target_population, I0 / target_population, R0, D0  # SEIRD IC array
    
    """
    What to run
    """
    has_to_run_sir = True
    has_to_run_sird = False
    has_to_run_seir = True
    has_to_run_seird = False
    has_to_run_seirdq = True

    
    
    
    
    
    
    return



main()
