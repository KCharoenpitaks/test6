# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:08:49 2020

@author: tanaj
module_2 is the correlation among issuer, within non-trading box.
"""

import pandas as pd
import numpy as np

class GetPivotForTradingBox:
    
    def __init__(self, filtered_input):
        
        self.col = ['cluster_id','box_id','symbol','is_traded_today','is_pivot','ttm','diff_static_spread']
        self.pivot_for_trading_box = pd.DataFrame(columns=self.col)
        self.filtered_input = filtered_input
        
        
    def get_pivot(self):
        
        col = self.col
        filtered_input = self.filtered_input
        temp = filtered_input[filtered_input['is_traded_today']==True]
        temp['ttm'] = temp['ttm'].astype(float)
        temp['volume_time_ttm'] = temp['ttm'].astype(float) * temp['total_volume'].astype(float)
        temp_1_ = temp.drop_duplicates(subset=['symbol', 'ttm'], keep='last')
        ###############################
        # Weighted with # of trade
        #temp_2 = temp.groupby(['box_id']).mean() # Change Here # of trade average for pivot
        ###############################
        # Weighted with Simple Average
        #temp_2 = temp_1_.groupby(['box_id']).mean() #Change base on simple average
        ###############################
        # Weighted with trade_volume
        temp_2 = temp_1_.groupby(['box_id']).mean()
        temp_2['ttm'] = temp_2['volume_time_ttm']/temp_2['total_volume']
        temp_2['box_id'] = temp_2.index.astype(int)
        temp_2['symbol'] = 'pivot_' + temp_2['box_id'].astype(str)
        temp_2['is_pivot'] = True
        temp_2['is_traded_today'] = True
        temp_3 = temp_2[col].reset_index(drop=True)
        self.pivot_for_trading_box = temp_3
     
    
class GetTradingBoxNonTradingIssuerPair:
    
    def __init__(self, input_data):
        
        self.trading_box_id_non_trading_issuer_pair = pd.DataFrame()
        self.input_data = input_data
        
    def get_trading_box_id_non_trading_issuer_pair(self):
        
        input_data = self.input_data
        
        # get trading issuers within a trading box
        temp = input_data[input_data['is_traded_today']==True][['box_id','issuer']].drop_duplicates()
        
        # get all issuers within a trading box
        temp_2 = input_data[input_data['box_id'].isin(temp['box_id'])][['box_id','issuer']].drop_duplicates()
        
        # get non-trading issuers within a trading_box
        temp_3 = temp_2[(temp_2['box_id'].isin(temp['box_id']))&(~temp_2['issuer'].isin(temp['issuer']))]
        
        self.trading_box_id_non_trading_issuer_pair = temp_3
        

class GetPivotMaster:
    
    def __init__(self, pivot_for_trading_box, pivot_for_nontrading_box):
        
        self.pivot_for_trading_box = pivot_for_trading_box
        self.pivot_for_nontrading_box = pivot_for_nontrading_box
        
    def overwrite_two_df_to_get_pivot_master(self):
        
        pivot_for_trading_box = self.pivot_for_trading_box
        pivot_for_nontrading_box = self.pivot_for_nontrading_box
        
        pivot_for_trading_box['to_box_id'] = pivot_for_trading_box['box_id']
        pivot_for_trading_box['from_box_id'] = np.nan
        pivot_for_trading_box['corr_factor'] = 1
        trading_box_id = list(pivot_for_trading_box['box_id'])
        
        pivot_for_nontrading_box['box_id'] = pivot_for_nontrading_box['to_box_id']
        pivot_for_nontrading_box = pivot_for_nontrading_box[~pivot_for_nontrading_box['box_id'].isin(trading_box_id)]
        
        pivot_master = pd.concat([pivot_for_trading_box, pivot_for_nontrading_box])
        
        self.pivot_master = pivot_master.sort_values('box_id')
        