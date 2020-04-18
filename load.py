# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:50:46 2020

Caricamento base dati

@author: SalatiSt
"""

import pandas as pd
import numpy as np
import datetime as dt


def load(keys):
    def extract_ita(tipo, code):
        if tipo == "nat":
            url = "C:\git_covid19\italia\COVID-19\dati-andamento-nazionale\dpc-covid19-ita-andamento-nazionale.csv"
            df = pd.read_csv(url)
        elif tipo == "reg":
            url = "C:\git_covid19\italia\COVID-19\dati-regioni\dpc-covid19-ita-regioni.csv"
            df_raw = pd.read_csv(url)
            is_code =  df_raw['codice_regione']==code
            df = df_raw[is_code]
        elif tipo == "prov":
            url = "C:\git_covid19\italia\COVID-19\dati-province\dpc-covid19-ita-province.csv"
            df_raw = pd.read_csv(url)
            is_code =  df_raw['codice_provincia']==code
            df = df_raw[is_code]
    
        # Interpretazione
        #print(df)
        try:
            df = df[['data', 'totale_casi', 'deceduti','dimessi_guariti']].copy()
        except:
            df = df[['data', 'totale_casi']].copy()
            df['deceduti'] = 0
            df['dimessi_guariti'] = 0
            
        #df = df.loc[:,['data','totale_casi','deceduti','dimessi_guariti']]
        date = df['data']
        FMT = '%Y-%m-%dT%H:%M:%S'
        df['date_n'] = pd.Series(date.map(lambda x : (dt.datetime.strptime(x, FMT) - dt.datetime.strptime("2019-12-31T00:00:00", FMT)).days  ), index=df.index)
        df.index = df['date_n']
    
        return df
    
    def extract_jhu(url):
        df = pd.read_csv(url, delimiter=',')
        df = df.transpose()
        
        # Create date_n
        date_n = list(df.index[4:].values)
        try:
            FMT = '%m/%d/%y'
            date_n = map(lambda x : (dt.datetime.strptime(x, FMT) - dt.datetime.strptime("12/31/19", FMT)).days, date_n)
        except:
            FMT = '%m/%d/%Y'
            date_n = map(lambda x : (dt.datetime.strptime(x, FMT) - dt.datetime.strptime("12/31/2019", FMT)).days, date_n)
        new_header = df.iloc[1]
        df.columns = new_header
        df = df[4:]
        df['date_n'] = pd.Series(date_n, index=df.index)
        
        # Format date with a standard format
        df['date'] = pd.Series(df.index, index=df.index)
        FMTout = "%Y-%m-%d"
        try:
            FMTin = "%m/%d/%y"
            #df['date'] = map(lambda x : (datetime.strftime(datetime.strptime(x, FMTin), FMTout)), pd.Series(df.index, index=df.index))
            df['date'] = list(dt.datetime.strftime(dt.datetime.strptime(x, FMTin), FMTout) for x in pd.Series(df.index, index=df.index))
        except:
            FMTin = "%m/%d/%Y"
            #df['date'] = map(lambda x : (datetime.strftime(datetime.strptime(x, FMTin), FMTout)), pd.Series(df.index, index=df.index))
            df['date'] = list(dt.datetime.strftime(dt.datetime.strptime(x, FMTin), FMTout) for x in pd.Series(df.index, index=df.index))
        
        # Index based on date_n        
        #df.index = range(len(df))
        df.index = df['date_n']
        
        # Sum country regions
        df = df.groupby(level=0, axis=1).sum()
        
        # Reorder columns to make in more readable
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]
        
        # Filter only countris I am interested in
        #df = df[['date_n','Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Austria', 'US', 'China']]
        
        return df
    
    def concatandrename(df, df2, column, name):
        df = pd.concat([df, df2[column]], axis=1)
        df = df.rename(columns={column: name})
        return df
        
    # Extract from JHU database
    url_jhu_cases = "C:\git_covid19\jhu\COVID-19\csse_covid_19_data\csse_covid_19_time_series\\time_series_covid19_confirmed_global.csv"
    url_jhu_deaths = "C:\git_covid19\jhu\COVID-19\csse_covid_19_data\csse_covid_19_time_series\\time_series_covid19_deaths_global.csv"
    url_jhu_recovered = "C:\git_covid19\jhu\COVID-19\csse_covid_19_data\csse_covid_19_time_series\\time_series_covid19_recovered_global.csv"
    df_cases = extract_jhu(url_jhu_cases)
    df_deaths = extract_jhu(url_jhu_deaths)
    df_recovered = extract_jhu(url_jhu_recovered)
    
    # Extract from Ita database
    df_ita = extract_ita("nat", -1)
    df_emiliaromagna = extract_ita("reg", 8)
    df_lombardia = extract_ita("reg", 3)
    df_veneto = extract_ita("reg", 5)
    df_parma = extract_ita("prov", 34)
    df_reggioemilia = extract_ita("prov", 35)
    df_modena = extract_ita("prov", 36)
    
    # Merge Ita data with JHU data
    df_cases = concatandrename(df_cases, df_ita, "totale_casi", "Italia")
    df_cases = concatandrename(df_cases, df_emiliaromagna, "totale_casi", "EmiliaRomagna")
    df_cases = concatandrename(df_cases, df_lombardia, "totale_casi", "Lombardia")
    df_cases = concatandrename(df_cases, df_veneto, "totale_casi", "Veneto")
    df_cases = concatandrename(df_cases, df_parma, "totale_casi", "Parma")
    df_cases = concatandrename(df_cases, df_reggioemilia, "totale_casi", "Reggio")
    df_cases = concatandrename(df_cases, df_modena, "totale_casi", "Modena")
    
    df_deaths = concatandrename(df_deaths, df_ita, "deceduti", "Italia")
    df_deaths = concatandrename(df_deaths, df_emiliaromagna, "deceduti", "EmiliaRomagna")
    df_deaths = concatandrename(df_deaths, df_lombardia, "deceduti", "Lombardia")
    df_deaths = concatandrename(df_deaths, df_veneto, "deceduti", "Veneto")
    df_deaths = concatandrename(df_deaths, df_parma, "deceduti", "Parma")
    df_deaths = concatandrename(df_deaths, df_reggioemilia, "deceduti", "Reggio")
    df_deaths = concatandrename(df_deaths, df_modena, "deceduti", "Modena")
    
    df_recovered = concatandrename(df_recovered, df_ita, "dimessi_guariti", "Italia")
    df_recovered = concatandrename(df_recovered, df_emiliaromagna, "dimessi_guariti", "EmiliaRomagna")
    df_recovered = concatandrename(df_recovered, df_lombardia, "dimessi_guariti", "Lombardia")
    df_recovered = concatandrename(df_recovered, df_veneto, "dimessi_guariti", "Veneto")
    df_recovered = concatandrename(df_recovered, df_parma, "dimessi_guariti", "Parma")
    df_recovered = concatandrename(df_recovered, df_reggioemilia, "dimessi_guariti", "Reggio")
    df_recovered = concatandrename(df_recovered, df_modena, "dimessi_guariti", "Modena")
    
    print("Elements in df_cases: {}".format(len(df_cases)))
    print("Elements in df_deaths: {}".format(len(df_deaths)))
    print("Elements in df_recovered: {}".format(len(df_recovered)))
    
    # Prepara dati per processamento
    if len(keys)>0:
        x = np.asarray(list(df_cases['date_n']))
        X = len(keys)*[x]
        Y_cases = [np.asarray(list(df_cases[key])) for key in keys]
        Y_deaths = [np.asarray(list(df_deaths[key])) for key in keys]
        Y_recovered = [np.asarray(list(df_recovered[key])) for key in keys]
        Label_cases = keys
        Label_deaths = [Label_cases[i]+"+" for i in range(len(Label_cases))]
        Label_recovered = [Label_cases[i]+"|" for i in range(len(Label_cases))]
    else:
        X = []
        Y = []
        Y_cases = []
        Y_deaths = []
        Y_recovered = []
        Label_cases = []
        Label_deaths = []
        Label_recovered = []
    
    return df_cases, df_deaths, df_recovered, X, Y_cases, Y_deaths, Y_recovered, Label_cases, Label_deaths, Label_recovered


def country(df_cases, df_deaths, df_recovered, name):
    df = pd.DataFrame(index=df_cases.index)
    df = pd.concat([df, df_cases['date'], df_cases[name]], axis=1)
    df = pd.concat([df, df_deaths[name]], axis=1)
    df = pd.concat([df, df_recovered[name]], axis=1)
    df.columns = ['date', 'confirmed', 'deaths', 'recovered']
    df['confirmed_marker'] = 'Confirmed'
    df['deaths_marker'] = 'Death'
    df['recovered_marker'] = 'Recovered'
    #df = df.reset_index(drop=True)
    return df

def resetcountryday0(df):
    df0 = df[df.confirmed > 0]
    #df0 = df0.reset_index(drop=False)
    #df0 = df0.rename(columns={"date_n": "day"})
    return df0