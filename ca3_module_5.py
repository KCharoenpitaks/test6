# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:12:52 2021

@author: tanaj
"""

from ca3_module_dataprep import *

class StandardizeTransactionData(DataPrepFunc, ConnectToDatabase, MinMaxScaler):
    
    def __init__(self, trade_transaction_data):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.trade_transaction_data = trade_transaction_data
        
    def execute_fetch_commit_close(self):

        query = """                
               WITH temp AS
                (
                    SELECT symbol,
                            previous_market_value,
                            counter_party,
                            volume,
                            ttm,
                            trade_type,
                            trade_yield,
                            new_issued
                    FROM ca_data_for_minmax_scaler
                )
                
                SELECT  *
                FROM temp;"""      
                
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
        
    def append_data(self):
        
        data = self.data
        col = ['symbol',
               'previous_market_value',
               'counter_party',
               'volume',
               'ttm',
               'trade_type',
               'trade_yield',
               'new_issued']
        data = pd.DataFrame(data, columns = col)
        trade_transaction_data = self.trade_transaction_data
        self.number_of_new_data = trade_transaction_data.shape[0]
        self.data_2 = pd.concat([data, self.trade_transaction_data], ignore_index=True)
        
    def do_minmax_scaler(self):
        
        data_2 = self.data_2
        col_one_hot = ['counter_party',
                       'trade_type']
        col_same = ['previous_market_value',
                    'volume',
                    'ttm',
                    'trade_yield',
                    'new_issued']
        temp = pd.get_dummies(data_2[col_one_hot])
        data_3 = pd.concat([data_2[col_same], temp], axis=1)
        
        scaler = self.scaler
        scaler.fit(data_3)
        temp_2 = scaler.transform(data_3)
        data_final = pd.DataFrame(temp_2, columns=data_3.columns)
        
        n = -1 * self.number_of_new_data
        self.df = data_final[n:]


class MakeANNPrediction():
    
    def __init__(self, scaled_input_data, date_today):
        
        self.scaled_input_data = scaled_input_data
        self.date_today= date_today
        
    def load_the_model(self):
        
        self.model = keras.models.load_model('nl_ann_model')
        
    def make_nl_ann_prediction(self):
        
        model = self.model
        scaled_input_data = self.scaled_input_data
        nl_adjusted_prediction = model.predict(scaled_input_data)
        nl_adjusted_prediction = nl_adjusted_prediction * 100
        self.prediction = [item[0] for item in nl_adjusted_prediction]
        
        
class MakeNLDF(DataPrepFunc):
    
    def __init__(self, data, n_rows_flag, nl_prediction, trading_data, m2m_data):
        
        super().__init__()
        self.data = data
        self.n_rows_flag = n_rows_flag
        self.prediction = nl_prediction
        self.trading_data = trading_data
        self.m2m_data = m2m_data
        
    def extract_symbol(self):
        
        data = self.data
        n_rows_flag = self.n_rows_flag
        temp = data[-1*n_rows_flag:]['symbol'].to_list()
        
        self.nl_symbol = temp
        
    def merge_symbol_and_prediction(self):
        
        prediction = self.prediction
        symbol = self.nl_symbol
        # create a zip object from two lists
        zipObj = zip(symbol, prediction)
        #create a dataframe from the list of the zip object
        nl_prediction_2 = pd.DataFrame(list(zipObj), 
                                       columns=['symbol','diff_static_spread'])
        nl_prediction_2['diff_static_spread'] = nl_prediction_2['diff_static_spread']
        self.nl_prediction_2 = nl_prediction_2
        
    def merge_nl_prediction_with_m2m_data(self):
        
        trading_data = self.trading_data
        m2m_data = self.m2m_data
        m2m_data = m2m_data.drop(['last_trade_date',
                                'days_since_last_trade',
                                'is_traded_within_5bd',
                                'static_spread',
                                'weight_average_yield',
                                'total_volume'
                                ], 1)
        #change column names
        aDict = {
                 'prev_static_spread':'prev_m2m_static_spread',
                 'prev_bd':'prev_business_day'
                }
        m2m_data = m2m_data.rename(columns=aDict)
        nl_prediction_2 = self.nl_prediction_2
        temp = nl_prediction_2.merge(m2m_data, left_on='symbol', right_on='symbol', how='left')
        temp['static_spread'] = temp['prev_m2m_static_spread'] + temp['diff_static_spread']
        temp['trade_date'] = temp['asof']
        #set is_traded_today = true
        temp['is_traded_today'] = True
        temp_2 = pd.concat([trading_data,temp], ignore_index=True)
        
        #make standard datetime
        temp_2['asof'] = self.make_std_datetime(temp_2, 'asof')
        temp_2['prev_business_day'] = self.make_std_datetime(temp_2, 'prev_business_day')
        
        self.trading_data_final = temp_2