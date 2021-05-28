# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:18:16 2020

@author: Tanaj
"""
import pandas as pd
import numpy as np
from cal_rho_by_thai_asset import asset_sector_avg_corr
import statistics as st
#For trade correlation, data layer will be cluster, sector, box, issuer and trade value.

def get_clustering_data():
    filepath = r'C:\users\tanaj\desktop\credit assessment code\code\df_results_output.csv'
    df = pd.read_csv(filepath)
    df = df.rename(columns = {'sector':'Sector Abbr'})
    return df

#def get_clustering_data():
#    import requests
#    from io import StringIO
#    orig_url= 'https://drive.google.com/file/d/1V1x5H1nVY1KC_sRU9YmpvmLxralDNRi2/view?usp=sharing'
#    file_id = orig_url.split('/')[-2]
#    dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
#    url = requests.get(dwn_url).text
#    csv_raw = StringIO(url)
#    df = pd.read_csv(csv_raw)
#    df = df.rename(columns = {'issue':'symbol','sector':'Sector Abbr'})
#    return df
# Asset data is from SET
# Mapping data is in dictionary; cluster, sector, box, issuer and asset value.

def cal_asset_corr(data):    

    rho_data_all = data
    size = rho_data_all.shape[0]
    rho_cov = pd.DataFrame(np.zeros((size,size)), columns=rho_data_all.index, index=rho_data_all.index)
    
    for row in rho_data_all.index:
        for col in rho_data_all.index:
            rho_cov.loc[row,col]=pearson_correlation(rho_data_all.loc[row,:].values,rho_data_all.loc[col,:].values,rho_data_all.shape[1])
    
    return rho_cov

def cal_asset_corr(data):    

    rho_data_all = data
    size = rho_data_all.shape[1]
    rho_cov = pd.DataFrame(np.zeros((size,size)), columns=rho_data_all.columns, index=rho_data_all.columns)
    
    for row in rho_data_all.columns:
        for col in rho_data_all.columns:
            rho_cov.loc[row,col]=pearson_correlation(rho_data_all[row],rho_data_all[col],rho_data_all.shape[0])
    
    return rho_cov

#    return (rho_cov.values.sum()-size)/(size*size-size)
def get_total_asset_data():
    
    filepath = r'G:\Job\IFRS 9\LGD\Rho\rho_data.xlsx'
    rho_data = pd.read_excel(filepath)
    
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
    
    e = e[fill]
    # do the first difference along the column
    f = e.iloc[:,1:]
    f = f.astype(float)
    f = f.diff(periods=1, axis=1)
    f = f.iloc[:,1:]
    f['Symbol'] = e['Symbol']
    
    return f


def process_asset_corr_data(data,layer):

    TA_year_col = ['TA_2014','TA_2015','TA_2016','TA_2017','TA_2018']
    data[TA_year_col] = data[TA_year_col].apply(pd.to_numeric)

    if layer in ['cluster_id','box_id','issuer','Symbol','Sector Abbr']:
        TA_year_col.append(layer)
        data = data[TA_year_col]
        return data.groupby([layer]).mean()
      
    else: return 'Error!'
    
def pearson_correlation(array_x, array_y, length):

    sum1,sum2,sum3 = 0,0,0
    x_bar = st.mean(array_x)
    y_bar = st.mean(array_y)
    for row_index in range(length):
        sum1 += (array_x[row_index]-x_bar) * (array_y[row_index]-y_bar)
        sum2 += (array_x[row_index]-x_bar)**2
        sum3 += (array_y[row_index]-y_bar)**2
    return sum1/((sum2*sum3)**0.5)

def mapping_sector_data():
    filepath = r'G:\Job\IFRS 9\LGD\Rho\SET_listed_companies.csv'
    sector_mapping = pd.read_csv(filepath)
    return sector_mapping[['Symbol','Sector Abbr']][sector_mapping['Sector Abbr']!='-']

def cal_trade_corr(data):
    size = data.shape[0]
    trade_corr_data = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            trade_corr_data[i,j]=data.iloc[i,0]/data.iloc[j,0]

    return pd.DataFrame(data=trade_corr_data,index=data.index,columns=data.index)

def get_trade_data():
    
    filepath = r'C:\Users\tanaj\Desktop\Credit Assessment Code\Code\trade_corr_data.xlsx'
    data = pd.read_excel(filepath)
    data['trade_date'] = data['trade_date'].astype('datetime64[ns]')
    data = data.rename(columns=({'symbol':'issue'}))
    return data

def process_trade_corr_data(data,layer):

    TA_year_col = ['diff_static_spread']
    data[TA_year_col] = data[TA_year_col].apply(pd.to_numeric)

    if layer in ['cluster_id','box_id','issuer','Symbol']:
        TA_year_col.append(layer)
        data = data[TA_year_col]
        return data.groupby([layer]).mean()
      
    else: return 'Error!'
    
def initialize_cov_matrix(issuer_list_DF):

    issuer_list_df = clustering_data[['issuer','Sector Abbr']].drop_duplicates()
    issuer_list = list(issuer_list_df['issuer'])
    size = issuer_list_df.shape[0]
    temp = np.zeros((size,size))
    temp[:,:] = np.nan
    rho_cov = pd.DataFrame(temp, columns=issuer_list, index=issuer_list)
    
    return rho_cov

def issuer_to_sector_mapping(aDF):
    
    aDF = aDF.rename(columns={'sector_abbr':'Sector Abbr'})
    aDF = aDF[['issuer','Sector Abbr']].drop_duplicates()
    aDF = aDF.set_index('issuer')
    
    return aDF.to_dict()['Sector Abbr']

def map_trade_corr_with_sector_corr(cov, trade_corr, asset_sector_avg_corr):
    
    cov = cov.combine_first(trade_corr)
    for item in cov[pd.isnull(cov.iloc[:,1])].index:
        try:
            print(1)
            cov.loc[item,:] = asset_sector_avg_corr[mapping[item]]
            cov.loc[:,item] = asset_sector_avg_corr[mapping[item]]
        except:
            cov.loc[item,:] = 0.21
            cov.loc[:,item] = 0.21
            
    return cov

#fill in cov missing values with avg_asset_corr
def fill_missing_cov(cov, avg_asset_corr):
    max_val = max(cov.columns)
    ind = [i for i in range(max_val+1)]
    
    df = pd.DataFrame(np.nan, columns=ind, index=ind)
    df = df.combine_first(cov)
    
    return df.fillna(avg_asset_corr)

#make cov diagonal adjustment = 1
def set_diag_value(df,val):
    df.values[[np.arange(df.shape[0])]*2] = val
    return df

#replace inf/-inf with 0
def replace_inf_with_zero(df):
    return df.replace([np.inf, -np.inf],0)

#final clean the cov matrix
def final_clean_cov(cov, avg_asset_corr,layer):

    if layer == 'box_id':
        cov = fill_missing_cov(cov, avg_asset_corr)
        cov = set_diag_value(cov,1)
        cov = replace_inf_with_zero(cov)
        return cov
    else:
        cov = set_diag_value(cov,1)
        cov = replace_inf_with_zero(cov)
        return cov

#everything for asset correlation
def cal_corr_matrix_main(layer):
    #layer can be symbol, issuer, box_id, cluster_id
    total_asset_data = get_total_asset_data()
    sector_mapping = mapping_sector_data()
    clustering_data = get_clustering_data()
    all_asset_data = pd.merge(clustering_data, total_asset_data, left_on='issuer', right_on='Symbol')
    all_asset_data = pd.merge(all_asset_data, sector_mapping, left_on='issuer', right_on='Symbol')
    final_asset_data = process_asset_corr_data(all_asset_data, layer)
    final_asset_data = final_asset_data.T
    asset_corr = cal_asset_corr(final_asset_data)
    
    #everything for trade calculation
    trade_data = get_trade_data()
    all_trade_data = clustering_data.merge(trade_data, left_on='issuer', right_on='issuer')
    final_trade_data = process_trade_corr_data(all_trade_data, layer)
    
    trade_corr = cal_trade_corr(final_trade_data)
    
    # map asset and trade correlation together
    cov = pd.DataFrame()
    mapping = issuer_to_sector_mapping(clustering_data)
    cov = map_trade_corr_with_sector_corr(cov, asset_corr, asset_sector_avg_corr)
    
    #final clean the cov matrix
    cov = final_clean_cov(cov, 0.21, layer)
    
    return cov

#everything for asset correlation
def cal_corr_matrix_main_notrade(layer):
    #layer can be symbol, issuer, box_id, cluster_id
    total_asset_data = get_total_asset_data()
    sector_mapping = mapping_sector_data()
    clustering_data = get_clustering_data()
    all_asset_data = clustering_data.merge(total_asset_data, left_on='issuer', right_on='Symbol')
    all_asset_data = all_asset_data.merge(sector_mapping, left_on = 'issuer',right_on='Symbol')
    final_asset_data = process_asset_corr_data(all_asset_data, layer)
    final_asset_data = final_asset_data.T
    asset_corr = cal_asset_corr(final_asset_data)
    
    #final clean the cov matrix
    cov = final_clean_cov(asset_corr, 0.21, layer)
    
    return cov

def limit_min_max_one(aDF):
    
    aDF = aDF.applymap(lambda x: 1 if x>1 else x)
    aDF = aDF.applymap(lambda x: -1 if x<-1 else x)
    
    return aDF

cov_issuer = cal_corr_matrix_main('issuer')
cov_box_id = cal_corr_matrix_main('box_id')

cov_box_id.to_csv(r'C:\users\tanaj\desktop\credit assessment code\code\cov_box_id.csv')
#cov_box_id = cal_corr_matrix_main_notrade('box_id')