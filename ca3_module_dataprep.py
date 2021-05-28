# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:16:15 2020

@author: tanaj
"""

#  All functions return pandas dataframe.
from datetime import datetime, timedelta
from ca3_module_1_no_feature import SharedDecayCorrData
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler

class DataPrepFunc:

    def make_std_datetime(self, df, aCol):
            
        try:
            fm = '%Y-%m-%d'
            df[aCol] = pd.to_datetime(df[aCol], format=fm)
        except ValueError:
            fm = '%d/%m/%Y'
            df[aCol] = pd.to_datetime(df[aCol], format=fm)
        return df


class ReadCovBoxID:
    
    def __init__(self):
        
        self.df = None
        
    def read_from_csv(self):
        
        filepath= r'./cov_box_id.csv'
        df = pd.read_csv(filepath)
        self.df = df

        
class ConnectToDatabase:
    
    def  __init__(self):
        
        self.conn = None
        self.cursor = None
        self.data = None
    
    def create_database_connection(self):
        
        Server = '192.168.70.40'
        Database = 'credit_assessment'
        uid = 'postgres'
        pwd = 'password'
        port = '5432'
        conn = psycopg2.connect(host=Server,
                               port=port,
                               user=uid,
                               password=pwd,
                               database=Database)
        self.conn = conn
        
    def get_cursor(self):
        
        cursor = self.conn.cursor()
        self.cursor = cursor
    
    def execute_fetch_commit_close(self):
        pass
    
    def make_df():
        pass


class GetPrevBD(ConnectToDatabase):
    
    def __init__(self, date):
        
        self.date = date
    
    def get_cursor(self):
        
        cursor = self.conn.cursor()
        self.cursor = cursor
    
    def execute_fetch_commit_close(self):
        
        query = """
                SELECT DISTINCT(asof)
                FROM ca_website_market_yield
                """
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
        
    def make_df(self):
        
        df = pd.DataFrame(self.data, columns=['asof'])
        df = df.sort_values('asof')
        df = df.reset_index(drop=True)
        list0 = df['asof'].tolist()
        list0 = [pd.Timestamp(item) for item in list0]
        self.list0 = list(dict.fromkeys(list0))
        
    def drop_month_end(self):
        
        date_to_drop_ls = ['2018-12-31',
                           '2019-03-31',
                           '2019-06-30',
                           '2019-08-31',
                           '2019-11-30',
                           '2019-12-31',
                           '2020-02-29',
                           '2020-05-31',
                           '2020-10-31',
                           '2020-12-31',
                           '2021-01-31',
                           '2021-02-28',
                           '2021-07-31',
                           '2021-10-31',
                           '2021-12-31']
    
        date_to_drop_ts = [pd.Timestamp(item) for item in date_to_drop_ls]
        self.list0 = [item for item in self.list0 if item not in date_to_drop_ts]
        
    def get_prev_bd_index(self):
        
        date = pd.Timestamp(self.date)
        temp = self.list0.index(date)
        self.today_index = temp
        self.prev_bd_index = temp - 1
        
    def get_prev_bd(self):
        
        self.date_prev_bd = str(self.list0[self.prev_bd_index])+'.000'
        self.date_today = str(self.list0[self.today_index])+'.000'
        

class ReadM2MData(DataPrepFunc, ConnectToDatabase):
    
    def __init__(self, date_prev_bd, date_today):
        super().__init__()
        self.date_prev_bd = date_prev_bd
        self.date_today = date_today
        
    def execute_fetch_commit_close(self):
        
        query = """ --get original market yield data
                    WITH temp AS 
                    (
                        SELECT a.asof,
                            a.symbol,
                            a.ttm,
                            a.static_spread
                        FROM ca_website_market_yield AS a
                        WHERE (a.asof::date BETWEEN '""" +  self.date_prev_bd + """' AND '""" + self.date_today + """') AND
                              (a.static_spread IS NOT NULL)
                    ),
                    
                    --get previous business days
                    temp2 AS
                    (
                        SELECT a.*,
                            MAX(b.asof) AS prev_business_day
                        FROM temp AS a
                        LEFT JOIN temp AS b
                        ON a.symbol = b.symbol
                            AND a.asof > b.asof
                        GROUP BY a.symbol,
                                 a.asof,
                                 a.ttm,
                                 a.static_spread
                    ),
                    
                    --get previous m2m static spread
                    temp3 AS
                    (
                        SELECT a.*,
                               b.static_spread AS prev_m2m_static_spread
                        FROM temp2 AS a
                        LEFT JOIN temp AS b
                        ON a.symbol = b.symbol
                            AND a.prev_business_day = b.asof
                    ),
                    
                    --get the last trade within 5 business days
                    temp4 AS
                    (
                        SELECT a.*,
                               MAX(b.asof) AS last_trade_date
                        FROM temp3 AS a
                        LEFT JOIN ca_pricing_trade_summary_report AS b
                        ON a.symbol = b.symbol
                            AND a.asof >= b.asof
                        GROUP BY a.asof,
                                a.prev_business_day,
                                a.prev_m2m_static_spread,
                                a.static_spread,
                                a.ttm,
                                a.symbol
                    ),
                    
                    --check if the last trade date is within 5 business days.
                    temp5 AS
                    (   SELECT a.*,
                                (a.asof::date - a.last_trade_date::date) AS days_since_last_trade,
                                CASE
                                    WHEN (a.asof::date - a.last_trade_date::date) <= 5
                                        THEN 1
                                        ELSE 0
                                END AS within_5_BDays
                        FROM temp4 AS a
                        WHERE a.asof::date = '""" + self.date_today + """'
                    ),
                    
                    --get weighted volume and static spread from trade summary report
                    temp6 AS
                    (
                        SELECT a.*,
                                b.static_spread,
                                b.weight_average_yield,
                                b.total_volume
                        FROM temp5 AS a
                        LEFT JOIN ca_pricing_trade_summary_report AS b
                        ON a.symbol = b.symbol
                            AND a.asof = b.asof
                    )
                    
                    SELECT *
                    FROM temp6;"""
                    
        
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
        
    def get_m2m_data(self):
        
        df = pd.DataFrame(self.data)
        col = {
                0:'asof',
                1:'symbol',
                2:'ttm',
                3:'static_spread',
                4:'prev_bd',
                5:'prev_static_spread',
                6:'last_trade_date',
                7:'days_since_last_trade',
                8:'is_traded_within_5bd',
                9:'static_spread',
                10:'weight_average_yield',
                11:'total_volume'
              }
        df = df.rename(columns=col)
        df = self.make_std_datetime(df, 'asof')
        self.df = df

        
class ReadClusteringData(DataPrepFunc, ConnectToDatabase):
    
    def __init__(self, date_today):
        super().__init__()
        self.date_today = date_today
        self.data = None
        self.df = None
        
    def execute_fetch_commit_close(self):
        
        query = """ 
                SELECT *
                FROM ca_box
                WHERE asof = '"""+ self.date_today + """'
                """
        
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
    
    def make_df(self):
        
        col = {
                0:'cluster_id',
                1:'box_id',
                2:'symbol',
                3:'issuer',
                4:'asof',
                5:'ttm',
                6:'rating',
                7:'sector_abbr'
                }
        df = pd.DataFrame(self.data)
        df = df.rename(columns=col)
        self.df = df
               
    
class GetSimT1Data(DataPrepFunc, ConnectToDatabase):
    
    def __init__(self, date_prev_bd, date_today):
        super().__init__()
        self.date_prev_bd = date_prev_bd
        self.date_today = date_today

    def execute_fetch_commit_close(self):
    
        query = """ WITH temp AS 
                    (
                        SELECT a.asof,
                            a.symbol,
                            a.ttm,
                            a.today_m2m_static_spread
                        FROM ca_simulationdb AS a
                        WHERE a.asof::date = '""" + self.date_prev_bd + """'

                    )
                    
                    SELECT *
                    FROM temp;"""
        
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
        
    def make_df(self):
        
        col = {
                '0':'asof',
                '1':'symbol',
                '2':'ttm',
                '3':'prev_m2m_static_spread'
                }
        df = pd.DataFrame(self.data, columns=col)
        df = df.rename(columns=col)
        self.df = df


class GetTradingSymbol(DataPrepFunc, ConnectToDatabase):
    
    def __init__(self, date_prev_bd, date_today):
        super().__init__()
        self.date_prev_bd = date_prev_bd
        self.date_today = date_today
        
    def execute_fetch_commit_close(self):

        query = """                
                --filter out all irrelevant trade transactions
                WITH temp2 AS
                (
                    SELECT a.asof,
                            a.symbol,
                            a.ttm
                    FROM ca_website_market_yield AS a
                    WHERE a.last_trade_date::date = '""" + self.date_today + """'
                        AND (a.symbol NOT LIKE 'CB%'
                        AND (a.symbol NOT LIKE 'TB%' OR a.symbol LIKE 'TBEV%')
                        AND a.symbol NOT LIKE 'LB%'
                        AND a.symbol NOT LIKE 'BOT%'
                        AND a.symbol NOT LIKE 'SB%'
                        AND a.symbol NOT LIKE '%PA'
                        AND a.symbol NOT LIKE 'ILB%')
                ),
                
                --put m2m static spread in
                temp3 AS
                (
                    SELECT a.*,
                            b.static_spread
                    FROM temp2 AS a
                    LEFT JOIN ca_website_market_yield AS b
                    ON a.asof = b.asof AND a.symbol = b.symbol
                )
                
                SELECT *
                FROM temp3;"""      
                
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
    
    def make_df(self):
        
        data = self.data
        df = pd.DataFrame(data, columns = ['asof','symbol','ttm','static_spread'])
        df = df.fillna(0)
        df['prev_business_day'] = self.date_prev_bd
        df['is_traded_today'] = True
        self.df = df


class GetSimDiffSpread:
    
    def __init__(self, date, c_data, m_data, s_t1_data, trading_symbol):
        
        self.date = date
        self.c_data = c_data
        self.m_data = m_data
        self.s_t1_data = s_t1_data
        self.trading_symbol = trading_symbol
    
    def join_all(self):
        
        date = self.date
        c_data = self.c_data
        m_data = self.m_data
        s_t1_data = self.s_t1_data
        
        trading_symbol = self.trading_symbol
        df = pd.merge(trading_symbol, c_data[['symbol','issuer']][c_data['asof']==date], on='symbol', how='left')
        df = pd.merge(df, s_t1_data[['symbol','prev_m2m_static_spread']], on='symbol', how='left')
        
        self.df = df
        
    def cal_diff_spread(self):
        
        df = self.df
        df['diff_static_spread'] = df['static_spread'].astype(float) - df['prev_m2m_static_spread'].astype(float)
        df = df[pd.notnull(df['prev_m2m_static_spread'])]
        self.df = df
        
    def misc_adj(self):
        
        df = self.df
        df['trade_date'] = df['asof']
        self.df = df
        
        
class GetTransactionData(DataPrepFunc, ConnectToDatabase, MinMaxScaler):
    
    def __init__(self, date_prev_bd, date_today):
        super().__init__()
        self.date_prev_bd = date_prev_bd
        self.date_today = date_today
        self.scaler = MinMaxScaler()
        
    def execute_fetch_commit_close(self):

        query = """                
               WITH temp AS
                (
                    SELECT *
                    FROM ca_trade_transaction_data
                    WHERE last_transaction = 'true'
                        AND bp_not_in_line = 'true'
                        AND reference_type = 'MTM'
                        AND attribute <> 'C'
                        AND switching = 'false'
                        AND big_lot = 'false'
                        AND small_lot = 'false'
                        AND is_government = 'false'
                        AND trade_by_price = 'false'
                        AND asof = '""" + self.date_today[:-4] + """'
                )
                
                SELECT  symbol, 
                        previous_market_value,
                        counter_party,
                        volume,
                        ttm,
                        trade_type,
                        trade_yield,
                        new_issued
                FROM temp
                ORDER BY asof;"""      
                
        cursor = self.cursor
        cursor.execute(query)
        data = cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        self.data = data
    
    def make_input_df(self):
        
        data = self.data
        col = ['symbol',
               'previous_market_value',
               'counter_party',
               'volume',
               'ttm',
               'trade_type',
               'trade_yield',
               'new_issued']
        
        data = pd.DataFrame(data, columns=col)
        
        self.df = data
              
        
class JoinAllData(SharedDecayCorrData):
    
    def __init__(self, date, c_data, t_data, m_data):
        
        self.df = None
        self.c_data = c_data
        self.t_data = t_data
        self.m_data = m_data
        self.date = date
    
    def get_master_data(self):
        
        # 'df' left joins tables and change the column names.
        clustering_data = self.c_data
        trading_data = self.t_data
        m2m_data = self.m_data
        
        df = pd.merge(m2m_data, trading_data, on='symbol', how='left')
        df = df.rename(columns={'asof_x':'asof', 'ttm_x':'ttm'})
        
        #create primary key column
        df['pm_key'] = df['symbol'] + df['asof'].astype(str).apply(lambda x: x[:6])
        clustering_data['pm_key'] = clustering_data['symbol'] + clustering_data['asof'].astype(str).apply(lambda x: x[:6])
        
        df = pd.merge(df, clustering_data, on='pm_key', how='left')
        
        d = {'asof':['asof','asof_y']}
        df = df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)
            
        # create the new field 'is_traded_today'
        # filter in only qualified transactions to 'is_traded_today'
        df = df.rename(columns={'asof_x':'asof', 'symbol_x':'symbol', 'ttm_x':'ttm',
                                'total_volume':'total_volume', 'weight_average_yield':'weighted_average_yield',
                                'static_spread_x':'static_spread_tradesum'})

        df['issuer'] = df['symbol'].str.extract(r'(^[A-Z]+)')
        
        # rearrange the column order
        print(df.columns)
        col = ['asof','cluster_id','box_id','symbol','issuer',
               'sector_abbr','rating','ttm','pm_key','is_traded_today',
               'trade_date','prev_bd','prev_m2m_static_spread',
               'diff_static_spread','is_traded_within_5bd', 'total_volume',
               'weighted_average_yield','static_spread_tradesum']

        df = df[col]
#        df = df.rename(columns={'ttm_x':'ttm'})
        
        #drop data that does not have 'box_id'.
        df = df[pd.notnull(df['box_id'])]
#        df = df[df['asof']==self.date]
        
        self.df = df

    def sort_values(self, aHead):
        return self.df.sort_values(aHead)

    def fill_in_missing_fields(self):
        
        temp = self.df
        temp_1 = list(temp.columns)
        temp_2 = SharedDecayCorrData.col
        add_col = np.setdiff1d(temp_2,temp_1)
        for item in add_col: temp[item] = np.nan
        self.filtered_input = temp[SharedDecayCorrData.col]
        
    def make_nontrading_bool_to_false(self):
        
        temp = self.df
        temp.loc[pd.isnull(temp['is_traded_today']),'is_traded_today'] = False
        self.df = temp
        
    def drop_3_added_columns(self):
        
        temp = self.df
        temp_2 = temp.drop(['total_volume','weighted_average_yield','static_spread_tradesum'], axis=1)
        self.df_2 = temp_2