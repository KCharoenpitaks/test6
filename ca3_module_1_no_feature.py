# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:09:05 2020

@author: tanaj
module_1 is the decay within issuer
"""

import numpy as np
from math import exp, log

import pandas as pd
from datetime import datetime

class SharedDecayCorrData:
    
    col = ['asof', 'cluster_id', 'box_id', 'symbol', 'issuer',
           'sector_abbr', 'rating', 'ttm', 'is_traded_today', 'is_pivot',
           'trade_date', 'prev_bd', 'prev_trade_static_spread',
           'diff_static_spread', 'adj_spread_pre', 'corr_factor',
           'adj_spread', 'status']
    
    def __init__(self, date, input_data, clustering_data, lamb):
        
        # general data
        self.date = date
        self.input = input_data
        self.lamb = lamb
        self.cluster_id = np.sort(clustering_data['cluster_id'].unique())
        self.box_id = np.sort(clustering_data['box_id'].unique())
        self.issuer = np.sort(clustering_data['issuer'].unique())
        self.col = ['asof', 'cluster_id', 'box_id', 'symbol', 'issuer',
                    'sector_abbr', 'rating', 'ttm', 'is_traded_today', 'is_traded_within_5bd',
                    'is_pivot', 'trade_date', 'prev_bd', 'prev_trade_static_spread',
                    'diff_static_spread', 'adj_spread_pre', 'corr_factor',
                    'adj_spread', 'status']
        
        # module 1
        self.trading_box_id_issuer_dict = None
        self.filtered_input = pd.DataFrame()
        self.decay_result = pd.DataFrame()
        
        # module 2
        self.pivot_for_trading_box = None
        self.box_id_trading_issuer_pair = None
        self.box_id_non_trading_issuer = None
        
        # module 3
        self.nontrading_box_id = None
        
                
class Filter(SharedDecayCorrData):
    
    def __init__(self, date, input_data, clustering_data, lamb):
        
        super().__init__(date, input_data, clustering_data, lamb)
        self.filtered_input = pd.DataFrame()
        self.trading_box_id_issuer_dict = {}
            
    def get_trading_box_id_issuer_dict(self):
        
        trading_box_id_issuer_dict = {}
        df = self.input
        box_id = self.box_id
        for b_id in box_id:
            a_filter = (df['box_id']==b_id) \
                        & (df['is_traded_today']==True)
            temp = list(set(df[a_filter]['box_id'].values))
            temp_2 = df[df['box_id'].isin(temp)]
            temp_3 = list(set(temp_2['issuer'].values))
#            print(temp)
            if len(temp) > 0: trading_box_id_issuer_dict[b_id] = temp_3
        self.trading_box_id_issuer_dict = trading_box_id_issuer_dict

    def filter_decay_data(self):

        df = self.input
        temp = self.filtered_input
        a_dict = self.trading_box_id_issuer_dict
        for key in a_dict.keys():
            for value in a_dict[key]:
                a_filter = (df['box_id']==key) & (df['issuer']==value)
                df_2 = df[a_filter]
                temp = pd.concat([temp, df_2])
        temp = temp.sort_values(['box_id','ttm'])   # This is a correct sorting. #A very poor design
        self.filtered_input = temp
        
    def fill_in_missing_fields(self):
        
        temp = self.filtered_input
        temp_1 = list(temp.columns)
        temp_2 = self.col
        add_col = np.setdiff1d(temp_2,temp_1)
        for item in add_col: temp[item] = np.nan
        self.filtered_input = temp[self.col]


class DecayFunction(SharedDecayCorrData):
    
    def __init__(self, date, input_data, clustering_data, lamb, filtered_input, trading_cluster_id_box_id_dict, pivot_master):
        
        super().__init__(date, input_data, clustering_data, lamb)
        self.input_data = input_data
        self.trading_cluster_id_box_id_dict = trading_cluster_id_box_id_dict
        self.pivot_master = pivot_master
        self.decay_result = pd.DataFrame()
    
    def decay(self):
        
        def limit_within_range(aNum, minN, maxN):
            return max(min(aNum, maxN), minN)
    
        def cal_decay_value(aDF, lamb, **kwargs):
            
            corr_factor = kwargs.get('c', None)
            
            pivot_ttm = float(aDF[aDF['is_traded_today']==True]['ttm'].values[0])
            pivot_diff_static_spread = float(aDF[aDF['is_traded_today']==True]['diff_static_spread'].values[0])
            
#            print('correlation factor is ' + str(corr_factor) +'. is a type of ' + str(type(corr_factor)))
            
            try:
                func = lambda ttm, lamb, corr_factor, pivot_diff_static_spread, pivot_ttm: (lamb * corr_factor * pivot_diff_static_spread)/(1+np.abs(float(ttm) - pivot_ttm))
                aDF['adj_spread'] = aDF['ttm'].apply(func, args=(lamb, corr_factor, pivot_diff_static_spread, pivot_ttm))
                aDF['adj_spread'] = aDF['adj_spread'].apply(limit_within_range, args=(-2,2))
            
            except: 
                # The line below is subject to error if the ttm of the two issues are duplicate.
                func = lambda ttm, lamb, pivot_diff_static_spread, pivot_ttm: (lamb * pivot_diff_static_spread)/(1+np.abs(float(ttm) - pivot_ttm))
                aDF['adj_spread'] = aDF['ttm'].apply(func, args=(lamb, pivot_diff_static_spread, pivot_ttm))
                aDF['adj_spread'] = aDF['adj_spread'].apply(limit_within_range, args=(-5,5))
                
            return aDF
            
        def linear_interpolation(aDF):
            
            aDF = aDF.set_index('ttm', drop=False)
            aDF['adj_spread'] = list(aDF['diff_static_spread'].interpolate())
            aDF = aDF.reset_index(drop=True)
            
            return aDF
              
        def drop_merge_duplicates(aDF):
            
            col = [item for item in list(aDF.columns) \
                   if (~('adj_spread' in item) or (item=='adj_spread'))]
            col.append('symbol')
            return aDF[col]
        
        def get_nontrading_index(aDF):
            
            aDF = aDF.reset_index(drop=True)
#            row = aDF.index[-aDF['is_traded_today']].to_list()
            row = aDF[aDF['is_traded_today']==False].index.tolist()
            return row
        
        def decay_wing_function(temp, temp_2, left_pivot_index, right_pivot_index, b_id, issuer, \
                                lamb, is_decay, trading_cluster_id_box_id_dict, **kwargs):
            
            corr_factor = kwargs.get('c', None)
            
            # get the decay wings
            left_decay_wing = pd.DataFrame()
            right_decay_wing = pd.DataFrame()
            
            #get the list of trading box_id
            temp_3 = trading_cluster_id_box_id_dict
            temp_3 = list(temp_3.values())
            temp_3 = np.concatenate(temp_3)
            
            # get left decay
#            print(temp)
            left_decay_wing = cal_decay_value(temp[:left_pivot_index+1], lamb, c=corr_factor)
            if (left_decay_wing.shape[0] > 1):
                row = get_nontrading_index(left_decay_wing)
                temp_col = left_decay_wing.columns.get_loc('status')
                if is_decay: left_decay_wing.iloc[row, temp_col] = 'decay_left'
                else:
                    if b_id in temp_3:
                        left_decay_wing.iloc[row, temp_col] = 'inbox_corr_left'
                    else:
                        left_decay_wing.iloc[row, temp_col] = 'outbox_corr_left'
#                # separating inbox/outbox correlation
                
            # get right decay
            right_decay_wing = cal_decay_value(temp[right_pivot_index:], lamb, c=corr_factor)
            if (right_decay_wing.shape[0] > 1):
                row = get_nontrading_index(right_decay_wing)
                temp_col = right_decay_wing.columns.get_loc('status')
                if is_decay: right_decay_wing.iloc[row, temp_col] = 'decay_right'
                else: 
#                # separating inbox/outbox correlation
                    if b_id in temp_3:
                        right_decay_wing.iloc[row, temp_col] = 'inbox_corr_right'
                    else:
                        right_decay_wing.iloc[row, temp_col] = 'outbox_corr_right'
                        
            # get mid decay
            mid_decay_wing = linear_interpolation(temp[left_pivot_index:right_pivot_index+1])
            mid_decay_wing = mid_decay_wing.iloc[1:mid_decay_wing.shape[0]-1,:]
            if (mid_decay_wing.shape[0] > 0):
                row = get_nontrading_index(mid_decay_wing)
                temp_col = mid_decay_wing.columns.get_loc('status')
                mid_decay_wing.iloc[row, temp_col] = 'decay_mid'
            
            temp_2 = pd.concat([left_decay_wing, mid_decay_wing, right_decay_wing]).drop_duplicates()
            
            return temp_2
        
        #The name pivot for trading box is misleading and should be changed to pivot for non trading box.
        pivot_master = self.pivot_master
        input_data = self.input_data
        input_data['adj_spread'] = np.nan
        input_data['status'] = np.nan
        col = self.col
        temp_2 = pd.DataFrame(columns=col)
        b_id_list = input_data['box_id'].unique()
        lamb = float(self.lamb)
        trading_cluster_id_box_id_dict = self.trading_cluster_id_box_id_dict
        
        for b_id in b_id_list:
#            print(b_id)
            issuer_list = input_data['issuer'][input_data['box_id']==b_id].unique()
            
            for issuer in issuer_list:
                
#                print(issuer)
                wing_filter = (input_data['box_id']==b_id) & (input_data['issuer']==issuer)
                temp = input_data[wing_filter].sort_values(['ttm'])
                
                #This paragraph is to not adjust spread of newly traded bonds.
                non_adjust_filter = (temp['is_traded_within_5bd']==True)
                temp_non_adjust = temp[non_adjust_filter]
                temp_non_adjust['status'] = 'non-adjusted'
                temp_non_adjust['adj_spread'] = 0
                
                adj_filter = (temp['is_traded_within_5bd']==False) | (temp['is_traded_today']==True)
                temp = temp[adj_filter]
                temp = temp.reset_index(drop=True)
                temp = temp[col]
                
                try: 
#                    b_id, issuer, lamb = 8, 'QH', 28
#                    pivot_index = temp[temp['is_traded_today']==True].index.tolist()
#                    # get the pivot index
#                    left_pivot_index = pivot_index[0]
#                    right_pivot_index = pivot_index[-1]
#                    
#                    is_decay = True
                    
#                    temp_3 = decay_wing_function(temp, temp_2, left_pivot_index, right_pivot_index, \
#                                                 b_id, issuer, lamb, is_decay, trading_cluster_id_box_id_dict)
                    temp_3 = temp
                    
                except:
#                    b_id, issuer, lamb = 0, 'BJC', 0.7
                    try:
#                        filtered_pivot = (pivot_master['box_id']==b_id)
#                        the_pivot = pivot_master[filtered_pivot]
#                        temp = pd.concat([temp,the_pivot])
#                        temp = temp.sort_values(['ttm'])
#                        temp = temp.reset_index(drop=True)
#                        pivot_index = temp[temp['is_traded_today']==True].index.tolist()
#                        
#                        left_pivot_index = pivot_index[0]
#                        right_pivot_index = pivot_index[-1]
#                        
#                        is_decay = False
#                        
#                        box_id_corr_factor_dict = dict(zip(pivot_master['box_id'], \
#                                                           pivot_master['corr_factor']))
#                        
#                        corr_factor = box_id_corr_factor_dict[b_id]
#                        temp_3 = decay_wing_function(temp, temp_2, left_pivot_index, right_pivot_index, \
#                                                     b_id, issuer, lamb, is_decay, trading_cluster_id_box_id_dict, \
#                                                     c=corr_factor)
                        temp_3 = temp
                    
                    except IndexError:
                        pass

                try:
#                    print(temp_3)
                    temp_2 = pd.concat([temp_2, temp_3, temp_non_adjust])
                    temp_2 = temp_2.sort_values('ttm')
                except UnboundLocalError:
                    pass
                
                temp_3 = pd.DataFrame()

        self.decay_result = temp_2[self.col]
        
    def adj_asof(self):
        decay_result = self.decay_result
        decay_result['asof'] = self.date
        self.decay_result = decay_result

        #change decay_left/decay_right to corr_left/corr_right
                    
                
# Pivots are for issuers with 'no trade'. 1)No trade for the issuer but has trade with the box
# 2) No trade for the issuer and no trade within the box