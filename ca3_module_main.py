# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:10:20 2020

@author: tanaj
"""

import pandas as pd
import numpy as np
from flask import Flask, request
import logging

from ca3_module_dataprep import *
from ca3_module_1 import *
from ca3_module_2 import *
from ca3_module_3 import *
from ca3_module_4 import *
from ca3_module_5 import *

from timeit import default_timer as timer

# warnings.filterwarnings("ignore") #for tomorrow test

#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------

start = timer()

date = '2021-05-19'
lamb = 0.7

a = ReadClusteringData(date)
a.create_database_connection()
a.get_cursor()
a.execute_fetch_commit_close()
a.make_df()
clustering_data = a.df

a_1 = ReadCovBoxID()
a_1.read_from_csv()
cov_box_id = a_1.df

b_0 = GetPrevBD(date)
b_0.create_database_connection()
b_0.get_cursor()
b_0.execute_fetch_commit_close()
b_0.make_df()
b_0.drop_month_end()
b_0.get_prev_bd_index()
b_0.get_prev_bd()
date_prev_bd = b_0.date_prev_bd
date_today = b_0.date_today    

c_0 = ReadM2MData(b_0.date_prev_bd, b_0.date_today)
c_0.create_database_connection()
c_0.get_cursor()
c_0.execute_fetch_commit_close()
c_0.get_m2m_data()
m2m_data = c_0.df

# get nl model
c_1 = GetTransactionData(b_0.date_prev_bd, b_0.date_today)
c_1.create_database_connection()
c_1.get_cursor()
c_1.execute_fetch_commit_close()
c_1.make_input_df()
trade_transaction_data = c_1.df

c_2 = StandardizeTransactionData(c_1.df)
c_2.create_database_connection()
c_2.get_cursor()
c_2.execute_fetch_commit_close()
c_2.append_data()
c_2.do_minmax_scaler()
scaled_input_data = c_2.df

c_3 = MakeANNPrediction(c_2.df, b_0.date_today)
c_3.load_the_model()
c_3.make_nl_ann_prediction()
nl_prediction = c_3.prediction

b_1 = GetSimT1Data(b_0.date_prev_bd, b_0.date_today)
b_1.create_database_connection()
b_1.get_cursor()
b_1.execute_fetch_commit_close()
b_1.make_df()
sim_t1_data = b_1.df

b_2 = GetTradingSymbol(b_0.date_prev_bd, b_0.date_today)
b_2.create_database_connection()
b_2.get_cursor()
b_2.execute_fetch_commit_close()
b_2.make_df()
trading_symbol = b_2.df #make this line b.df

b_4 = GetSimDiffSpread(date, a.df, c_0.df, b_1.df, b_2.df)
b_4.join_all()
b_4.cal_diff_spread()
b_4.misc_adj()
trading_data = b_4.df

c_4 = MakeNLDF(c_2.data_2, c_2.number_of_new_data, c_3.prediction, b_4.df, c_0.df)
c_4.extract_symbol()
c_4.merge_symbol_and_prediction()
c_4.merge_nl_prediction_with_m2m_data()
trading_data_final = c_4.trading_data_final

d = JoinAllData(date, a.df, c_4.trading_data_final, c_0.df)
d.get_master_data()
d.sort_values('box_id')
d.fill_in_missing_fields()
d.make_nontrading_bool_to_false()
input_data = d.df
d.drop_3_added_columns()
input_data_drop = d.df_2

e = Filter(date, input_data, clustering_data, lamb)
e.get_trading_box_id_issuer_dict()
e.filter_decay_data()
e.fill_in_missing_fields()
filtered_input = e.filtered_input

f = GetTradingBoxNonTradingIssuerPair(input_data)
f.get_trading_box_id_non_trading_issuer_pair()
trading_box_id_non_trading_issuer_pair = f.trading_box_id_non_trading_issuer_pair

g = GetPivotForTradingBox(e.filtered_input)
g.get_pivot()
pivot_for_trading_box = g.pivot_for_trading_box

h = GetTradingDict(filtered_input, clustering_data)
h.get_trading_cluster_id_box_id_dict()
h.get_cluster_id_box_id_dict()
h.get_nontrading_cluster_id_box_id_dict()
trading_cluster_id_box_id_dict = h.trading_cluster_id_box_id_dict
cluster_id_box_id_dict = h.cluster_id_box_id_dict
nontrading_cluster_id_box_id_dict = h.nontrading_cluster_id_box_id_dict

i = GetMaxCorrForNonTradingBox(clustering_data, cluster_id_box_id_dict, 
                               trading_cluster_id_box_id_dict, 
                               nontrading_cluster_id_box_id_dict,
                               cov_box_id)
i.get_max_corr_for_each_box_id()
pivot_corr = i.pivot_corr
pivot_corr = pivot_corr.rename(columns={'corr':'corr_factor'})

j = GetPivotForNonTradingBox(pivot_corr, pivot_for_trading_box)
j.get_pivot_master()
j.misc_adj()
pivot_for_nontrading_box = j.pivot_for_nontrading_box
pivot_for_nontrading_box = pivot_for_nontrading_box[pd.notnull(pivot_for_nontrading_box['corr_factor'])]

k = GetPivotMaster(pivot_for_trading_box, pivot_for_nontrading_box)
k.overwrite_two_df_to_get_pivot_master()
pivot_master = k.pivot_master

l = DecayFunction(date, input_data_drop, clustering_data, lamb,
                  input_data, trading_cluster_id_box_id_dict,
                  pivot_master)
l.decay() # continue here
l.adj_asof()
decay_result = l.decay_result

m = AdjDataType(decay_result, SharedDecayCorrData.col)
m.adj_prevbd_type()
m.adj_tradedate_type()
decay_result = m.decay_result

n = MiscAdj(decay_result)
n.make_adj_spread_a_diff_static_spread()
n.replace_1_with_True_nan_with_false()
n.replace_nan_with_pivot_b_id()
n.make_today_trade_status_true()
n.make_adj_spread_for_nan_status_zero()
decay_result_final = n.decay_result

end = timer()
print('{} {} {} {:.2f} {}'.format('Run CA model', str(date), 'successful. Spent', end - start, 'seconds.'))

#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------