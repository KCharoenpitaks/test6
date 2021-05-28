# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:19:43 2020

@author: tanaj
"""

# get key for a certain box_id
import pandas as pd
import numpy as np

class AdjDataType():
    
    def __init__(self, decay_result, col):
        self.decay_result = decay_result
        self.col = col
        
    def adj_prevbd_type(self):
        decay_result = self.decay_result
        decay_result['prev_bd'] = decay_result['prev_bd'].astype('str')
        self.decay_result = decay_result[self.col]
        
    def adj_tradedate_type(self):
        decay_result = self.decay_result
        decay_result['trade_date'] = decay_result['trade_date'].astype('str')
        self.decay_result = decay_result[self.col]
        
    
class MiscAdj():
    
    def __init__(self, decay_result):
        self.decay_result = decay_result
        
    def make_adj_spread_a_diff_static_spread(self):
        decay_result =self.decay_result
        decay_result['adj_spread'][decay_result['is_traded_today']==True] = \
            decay_result['diff_static_spread'][decay_result['is_traded_today']==True] 
        self.decay_result = decay_result
        
    def replace_1_with_True_nan_with_false(self):
        decay_result = self.decay_result
        decay_result['is_pivot'] = decay_result['is_pivot'].replace(1, True)
        decay_result['is_pivot'] = decay_result['is_pivot'].replace(np.nan, False)
        self.decay_result = decay_result
        
    def replace_nan_with_pivot_b_id(self):
        decay_result = self.decay_result
        decay_result['status'][decay_result['is_pivot']] = decay_result['symbol'][decay_result['is_pivot']]
        self.decay_result = decay_result
        
    def make_today_trade_status_true(self):
        decay_result = self.decay_result
        decay_result['status'][decay_result['is_traded_today']] = 'traded_today'
        self.decay_result = decay_result
        
    def make_adj_spread_for_nan_status_zero(self):
        decay_result = self.decay_result
        decay_result['adj_spread'][pd.isnull(decay_result['status'])] = 0
        self.decay_result = decay_result
        