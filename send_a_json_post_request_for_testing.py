# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:21:46 2020

@author: tanaj
"""

import requests
import pandas as pd

date = '2019-01-09'
lamb = 0.7

response = requests.post(r'http://192.168.10.151:5000/ca_3/post', json={'asof_date': date, 'lambda': lamb})

print("Status code: ", response.status_code)

df = pd.DataFrame(response.json())
df
                                    
#get request
#http://192.168.10.151:5000/ca_3/web?asof_date=2019-03-05&lambda=0.7

df = pd.read_html(r'http://192.168.10.151:5000/ca_3/web?asof_date=2019-03-04&lambda=0.7')[0]


####==========================================================================

import psycopg2

host = '192.168.70.40'
port = '5432'
database = 'credit_assessment'
user = 'postgres'
password = 'password'

conn = psycopg2.connect(host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password)

sql_query = """
            SELECT DISTINCT(asof)
            FROM ca_website_market_yield
            """

cursor = conn.cursor()
cursor.execute(sql_query)
cursor.fetchall()
