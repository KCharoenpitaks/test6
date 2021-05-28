# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:31:08 2019

@author: Tanaj
"""

import numpy as np
import pandas as pd

#use the old rho
rho_data = pd.read_excel(r'G:\Job\IFRS 9\LGD\Rho\rho_data.xlsx')
rho_data = rho_data.rename(columns={"Total Asset  ('000 Baht)":'Total Asset',
                                    'Fiscal Year ':'Fiscal Year',
                                    'SYMBOL':'Symbol'})

rho_data_2013 = rho_data[rho_data['Fiscal Year']==2013][['Symbol','Total Asset']]
rho_data_2014 = rho_data[rho_data['Fiscal Year']==2014][['Symbol','Total Asset']]
rho_data_2015 = rho_data[rho_data['Fiscal Year']==2015][['Symbol','Total Asset']]
rho_data_2016 = rho_data[rho_data['Fiscal Year']==2016][['Symbol','Total Asset']]
rho_data_2017 = rho_data[rho_data['Fiscal Year']==2017][['Symbol','Total Asset']]
rho_data_2018 = rho_data[rho_data['Fiscal Year']==2018][['Symbol','Total Asset']]

a = pd.merge(rho_data_2013,rho_data_2014, on='Symbol', how='inner')
b = pd.merge(a, rho_data_2015, on='Symbol', how='inner')
c = pd.merge(b, rho_data_2016, on='Symbol', how='inner')
d = pd.merge(c, rho_data_2017, on='Symbol', how='inner')
e = pd.merge(d, rho_data_2018, on='Symbol', how='inner')

e.columns = ['Symbol','TA_2013','TA_2014','TA_2015','TA_2016','TA_2017','TA_2018']
fill = (e['TA_2013']!='-') & (e['TA_2014']!='-') & (e['TA_2015']!='-') & (e['TA_2016']!='-') & (e['TA_2017']!='-') & (e['TA_2018']!='-')

f = e[fill]

rho_data_all = e[fill].transpose()
rho_data_all.columns = rho_data_all.iloc[0]
rho_data_all = (rho_data_all.iloc[1:len(rho_data_all)]).astype(float)

size = rho_data_all.shape[1]
rho_cov = pd.DataFrame(np.zeros((size,size)), columns=rho_data_all.columns, index=rho_data_all.columns)

#Implement pearson correlation function
import statistics as st
def pearson_correlation(array_x, array_y, length):
    sum1,sum2,sum3 = 0,0,0
    x_bar = st.mean(array_x)
    y_bar = st.mean(array_y)
    for row_index in range(length):
        sum1 += (array_x[row_index]-x_bar) * (array_y[row_index]-y_bar)
        sum2 += (array_x[row_index]-x_bar)**2
        sum3 += (array_y[row_index]-y_bar)**2
    return sum1/((sum2*sum3)**0.5)

#Calculate rho in each cell of a dataframe
#This one is for LGD.
#for row in rho_data_all.columns:
#    for col in rho_data_all.columns:
#        rho_cov.loc[row,col]=pearson_correlation(rho_data_all[row],rho_data_all[col],rho_data_all.shape[0])
#
#finalized_rho = (rho_cov.values.sum()-size)/(size*size-size)

#==============================================================================
#This one is for correlation in credit assessment.
filepath = r'G:\Job\IFRS 9\LGD\Rho\SET_listed_companies.csv'
sector_mapping = pd.read_csv(filepath)
sector_data = sector_mapping[['Symbol','Sector Abbr']][sector_mapping['Symbol'].isin(rho_data_all.columns)]
temp_company = []
temp_sector = []
for sector in sector_data['Sector Abbr'].unique():
    temp_sector.append(sector)
    temp_company.append(list(sector_data[sector_data['Sector Abbr']==sector]['Symbol']))

#get sector dict
sector_dict = dict(zip(temp_sector,temp_company))

def cal_asset_correlation(data,aDict):    

    rho_data_all = data   
    size = rho_data_all.shape[1]
    rho_cov = pd.DataFrame(np.zeros((size,size)), columns=rho_data_all.columns, index=rho_data_all.columns)
    finalized_rho = []
    
    for key,value in sector_dict.items():
        size = rho_data_all[value].shape[1]
        rho_cov = pd.DataFrame(np.zeros((size,size)), columns=rho_data_all[value].columns, index=rho_data_all[value].columns)
        for row in rho_data_all[value].columns:
            for col in rho_data_all[value].columns:
                rho_cov.loc[row,col]=pearson_correlation(rho_data_all[row],rho_data_all[col],rho_data_all[value].shape[0])
        print(key)
        finalized_rho.append((rho_cov.values.sum()-size)/(size*size-size))
                
    return dict(zip(temp_sector,finalized_rho))

asset_sector_avg_corr = cal_asset_correlation(rho_data_all, sector_dict)
asset_box_avg_corr = cal_asset_correlation(rho_data_all, sector_dict)