# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:09:24 2020

@author: tanaj
module_3 is the correlation among box, same cluster.
"""

import pandas as pd

class GetTradingDict:
    
    def __init__(self, filtered_input, clustering_data):
        self.filtered_input = filtered_input
        self.trading_cluster_id_box_id_dict = {}
        self.clustering_data = clustering_data
        self.cluster_id_box_id_df = pd.DataFrame()
        self.cluster_id_box_id_dict = pd.DataFrame()
        self.nontrading_cluster_id_box_id_dict = {}
        
    def get_trading_cluster_id_box_id_dict(self):
        
        filtered_input = self.filtered_input
        temp = filtered_input[filtered_input['is_traded_today']==True][['cluster_id','box_id']].drop_duplicates()
        temp = temp.astype('int')
        self.trading_cluster_id_box_id_df = temp
        temp_3 = {}
        for item in temp['cluster_id'].unique():
            temp_2 = list(temp[temp['cluster_id']==item]['box_id'])
            temp_3[int(item)]=temp_2
            
        self.trading_cluster_id_box_id_dict = temp_3
        
    def get_cluster_id_box_id_dict(self):
        
        clustering_data = self.clustering_data
        temp = clustering_data[['cluster_id','box_id']].drop_duplicates()
        temp = temp.astype('int')
        self.trading_cluster_id_box_id_df = temp
        temp_3 = {}
        for item in temp['cluster_id'].unique():
            temp_2 = list(temp[temp['cluster_id']==item]['box_id'])
            temp_3[int(item)]=temp_2
            
        self.cluster_id_box_id_dict = temp_3
        
    def get_nontrading_cluster_id_box_id_dict(self):
        
        temp_1 = self.cluster_id_box_id_dict
        temp_2 = self.trading_cluster_id_box_id_dict
        temp_3 = {}
        for item in list(temp_1.keys()):
            try:
                temp_3[item] = [item_2 for item_2  in temp_1[item] if item_2 not in temp_2[item]]
            except KeyError:
                temp_3[item] = temp_1[item]
        
        self.nontrading_cluster_id_box_id_dict = temp_3
        
class GetMaxCorrForNonTradingBox:
    
    def __init__(self, clustering_data, cluster_id_box_id_dict,
                 trading_cluster_id_box_id_dict, 
                 nontrading_cluster_id_box_id_dict,
                 cov_box_id):
    
        self.pivot_corr = pd.DataFrame(columns=['to_box_id','from_box_id','corr_factor'])
        self.clustering_data = clustering_data
        self.cluster_id_box_id_dict = cluster_id_box_id_dict
        self.trading_cluster_id_box_id_dict = trading_cluster_id_box_id_dict
        self.nontrading_cluster_id_box_id_dict = nontrading_cluster_id_box_id_dict
        self.cov_box_id = cov_box_id
    
    def get_max_corr_for_each_box_id(self):
            
        # get key for a certain box_id
        def get_box_id_cluster_id_dict(clustering_data):
                
            aDf = clustering_data.set_index('box_id')['cluster_id']
            aDict = aDf.to_dict()
            
            return aDict
    
        def transpose(l1): 

            l2 =[[row[i] for row in l1] for i in range(len(l1[0]))] 
            
            return l2 
        
        def limit_min_max_one(aDF):
    
            aDF = aDF.applymap(lambda x: 1 if x>1 else x)
            aDF = aDF.applymap(lambda x: -1 if x<-1 else x)
    
            return aDF
        
        temp_df = pd.DataFrame()
        temp = []
        clustering_data = self.clustering_data
        nontrading_cluster_id_box_id_dict = self.nontrading_cluster_id_box_id_dict
        trading_cluster_id_box_id_dict = self.trading_cluster_id_box_id_dict
        cov_box_id = self.cov_box_id
        
        cov_box_id = limit_min_max_one(cov_box_id)
        aDict = get_box_id_cluster_id_dict(clustering_data) #self.clustering_data
        
        for key in nontrading_cluster_id_box_id_dict.keys():
            for item_2 in nontrading_cluster_id_box_id_dict[key]:
                try:
                    for item_3 in self.trading_cluster_id_box_id_dict[key]:
                        try:
                            temp.append([item_2,item_3,cov_box_id.iloc[item_2, item_3]])
                        except IndexError:
                            temp.append([item_2,item_3,0])
                except KeyError: pass
            
        temp_df['to_box_id'] = transpose(temp)[0]
        temp_df['from_box_id'] = transpose(temp)[1]
        temp_df['corr_factor'] = transpose(temp)[2]
        temp_df = temp_df.drop_duplicates()
        temp_df[temp_df['to_box_id']!=temp_df['from_box_id']]
        
        temp_df_2 = temp_df.groupby(['to_box_id'])
        temp_df_2 = temp_df_2.max()
        temp_df_2 = temp_df_2.reset_index()
        temp_df_2 = temp_df_2.rename(columns={'corr':'corr_factor'})
        
        self.pivot_corr = temp_df_2
        
    
class GetPivotForNonTradingBox:
    
    def __init__(self, pivot_corr, pivot_for_trading_box):
        
        self.pivot_corr = pivot_corr
        self.pivot_for_trading_box = pivot_for_trading_box
        self.pivot_for_nontrading_box = None
        self.pivot_master = None
        
    def get_pivot_master(self):
        
        temp = pd.merge(self.pivot_corr, self.pivot_for_trading_box, left_on='from_box_id', right_on='box_id', how='outer')
        
        # This line is to put a half effect on diff_static_spread
        temp['diff_static_spread'] = temp['diff_static_spread']*0.5
        
        self.pivot_for_nontrading_box = temp
        temp['to_box_id'] = temp['to_box_id'].fillna(temp['box_id'])
        temp['to_box_id'] = temp['to_box_id'].astype(int)
        temp['symbol'] = 'pivot_' + temp['to_box_id'].astype(str)
        temp = temp.drop(columns=['box_id','from_box_id'])
        temp = temp.rename(columns={'to_box_id':'box_id'})
        self.pivot_master = temp
        
    def misc_adj(self):
        
        pivot_master = self.pivot_master
        pivot_master = pivot_master[pd.notnull(pivot_master['corr_factor'])]
        self.pivot_mater = pivot_master.fillna(1)