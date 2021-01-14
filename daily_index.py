import pymongo
import pandas as pd
import numpy as np
import pickle
import datetime
import time
import gzip
import lzma
import pyTSL
import os

def DB(host, db_name, user, passwd):
    auth_db = db_name if user not in ('admin', 'root') else 'admin'
    url = 'mongodb://%s:%s@%s/?authSource=%s' % (user, passwd, host, auth_db)
    client = pymongo.MongoClient(url, maxPoolSize=None)
    db = client[db_name]
    return db

def read_memb_daily(db, name, start_date=None, end_date=None, skey=None, index_id=None, interval=None, col=None, return_sdi=True):
    collection = db[name]
    # Build projection
    prj = {'_id': 0}
    if col is not None:
        if return_sdi:
            col = ['skey', 'date', 'interval'] + col
        for col_name in col:
            prj[col_name] = 1

    # Build query
    query = {}
    if skey is not None:
        query['skey'] = {'$in': skey}
    if index_id is not None:
        query['index_id'] = {'$in': index_id}
    if interval is not None:
        query['interval'] = {'$in': interval}
    if start_date is not None:
        if end_date is not None:
            query['date'] = {'$gte': start_date, '$lte': end_date}
        else:
            query['date'] = {'$gte': start_date}
    elif end_date is not None:
        query['date'] = {'$lte': end_date}

    # Load data
    cur = collection.find(query, prj)
    df = pd.DataFrame.from_records(cur)
    if df.empty:
        df = pd.DataFrame()
    else:
        df = df.sort_values(by=['date', 'index_id', 'skey'])
    return df

def build_query(start_date=None, end_date=None, index_id=None):
    query = {}
    def parse_date(x):
        if type(x) == int:
            return x
        elif type(x) == str:
            if len(x) != 8:
                raise Exception("date must be YYYYMMDD format")
            return int(x)
        elif type(x) == datetime.datetime or type(x) == datetime.date:
            return x.strftime("%Y%m%d").astype(int)
        else:
            raise Exception("invalid date type: " + str(type(x)))
    if start_date is not None or end_date is not None:
        query['date'] = {}
        if start_date is not None:
            query['date']['$gte'] = parse_date(start_date)
        if end_date is not None:
            query['date']['$lte'] = parse_date(end_date)
    def parse_symbol(x):
        if type(x) == int:
            return x
        else:
            return int(x)
    if index_id:
        if type(index_id) == list or type(index_id) == tuple:
            query['index_id'] = {'$in': [parse_symbol(x) for x in index_id]}
        else:
            query['index_id'] = parse_symbol(index_id)
    return query

def write_memb_data(db, name, df):
    collection = db[name]
    df1 = []
    for symbol in df['index_id'].unique():
        if symbol in collection.distinct('index_id'):
            symbol = int(symbol)
            m_ax = pd.DataFrame.from_records(collection.find({'index_id':{'$in':[symbol]}}).sort([('date',-1)]).skip(0).limit(1))['date'].values[0]
            df2 = df[(df['index_id'] == symbol) & (df['date'] > m_ax)]
            print(df2)
            df1 += [df2]
        else:
            print(symbol)
            df2 = df[(df['index_id'] == symbol)]
            print(df2)
            df1 += [df2]
    df1 = pd.concat(df1).reset_index(drop=True)
    df1 = df1.to_dict('records')
    collection.insert_many(df1)

def delete_memb_data(db, name, start_date=None, end_date=None, index_id=None):
    collection = db[name]
    query = build_query(start_date, end_date, index_id)
    if not query:
        print('cannot delete the whole table')
        return None
    collection.delete_many(query)

database_name = 'com_md_eq_cn'
user = "zhenyuy"
password = "bnONBrzSMGoE"

pd.set_option('max_columns', 200)
db1 = DB("192.168.10.178", database_name, user, password)

import sys
sys.path.append('C:\\Program Files\\Tinysoft\\Analyse.NET')
import TSLPy3
TSLPy3.ConnectServer("tsl.tinysoft.com.cn",443)
dl = TSLPy3.LoginServer("jqtz","+7.1q2w3e")
assert(dl[0] == 0)

import pymongo
import pandas as pd
import numpy as np
import pickle
import datetime
import time
import gzip
import lzma
import pytz

startDate = datetime.datetime.today().strftime('%Y%m%d')
endDate = datetime.datetime.today().strftime('%Y%m%d')

def download_index(startDate, endDate, indexCode):
    tsstr = """
               indexTicker:= '{}';
               BegT:= {};
               EndT:= {} + 0.99;
               dateArr:=MarketTradeDayQk(BegT,EndT);
               r:=array();
               for nI:=0 to length(dateArr)-1 do
               begin
                 GetBKWeightByDate(indexTicker,dateArr[nI],t);
                 t := t[:,array("截止日","代码","比例(%)")]; 
                 r:=r union t;
               end;
               return r;  
            """.format(indexCode, startDate + 'T', endDate + 'T')
    # weight_table = pd.DataFrame(c.exec(tsstr).value())
    weight_table = pd.DataFrame(TSLPy3.RemoteExecute(tsstr,[],{})[1])
    weight_table.columns=['date','weight','ID']
    weight_table['ID'] = weight_table['ID'].str.decode('GBK')
    weight_table['date'] = pd.to_datetime(weight_table.date.astype(str))
    return weight_table
IF_weight = download_index(startDate, endDate, 'SH000300')
IC_weight = download_index(startDate, endDate, 'SH000905')
CSI1000_weight = download_index(startDate, endDate, 'SH000852')
weight_table = download_index(startDate, endDate, 'SH000985')

CSIRest_weight = []
for day in weight_table.date.unique():
    IC_stock = list(IC_weight[IC_weight.date == day].ID.unique())
    IF_stock = list(IF_weight[IF_weight.date == day].ID.unique())
    CSI1000_stock = list(CSI1000_weight[CSI1000_weight.date == day].ID.unique())
    ex_stock = list(set(IC_stock + IF_stock + CSI1000_stock))
    assert len(ex_stock) == 1800
    CSIRest_weight_day = weight_table[(weight_table.date == day) & (~weight_table.ID.isin(ex_stock))]
    CSIRest_weight += [CSIRest_weight_day]
CSIRest_weight = pd.concat(CSIRest_weight).reset_index(drop=True)
sumWeightToday = CSIRest_weight.groupby('date')['weight'].sum().reset_index()
sumWeightToday.rename(columns = {'weight':'sumWeightDay'}, inplace = True)
weight_table = CSIRest_weight.merge(sumWeightToday, on = 'date', how = 'left')
weight_table['weight'] = weight_table['weight'] / weight_table['sumWeightDay'] * 100
weight_table = weight_table.drop(columns = {'sumWeightDay'})

import pymongo
import pandas as pd
import numpy as np
import pickle
import datetime
import time
import gzip
import lzma
import pytz
import TSLPy3

IF_weight['date'] = IF_weight['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
IF_weight['ID'] = np.where(IF_weight['ID'].str[:2] =='SZ', IF_weight['ID'].str[2:].astype(int) + 2000000, IF_weight['ID'].str[2:].astype(int) + 1000000)
IF_weight = IF_weight.rename(columns={'ID':'skey'})
IF_weight['index_id'] = 1000300
IF_weight['index_name'] = 'IF'
IF_weight = IF_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = IF_weight.groupby('date')['weight'].sum().reset_index()
assert(k[k['weight'] - 100 > 0.02].shape[0] == 0)
write_memb_data(db1, 'index_memb', IF_weight)
print(IF_weight)
os.mkdir('E:\\daily_index_result\\' + startDate)
IF_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\IF.csv')

IC_weight['date'] = IC_weight['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
IC_weight['ID'] = np.where(IC_weight['ID'].str[:2] =='SZ', IC_weight['ID'].str[2:].astype(int) + 2000000, IC_weight['ID'].str[2:].astype(int) + 1000000)
IC_weight = IC_weight.rename(columns={'ID':'skey'})
IC_weight['index_id'] = 1000905
IC_weight['index_name'] = 'IC'
IC_weight = IC_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = IC_weight.groupby('date')['weight'].sum().reset_index()
assert(k[k['weight'] - 100 > 0.02].shape[0] == 0)
write_memb_data(db1, 'index_memb', IC_weight)
print(IC_weight)
IC_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\IC.csv')

CSI1000_weight['date'] = CSI1000_weight['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
CSI1000_weight['ID'] = np.where(CSI1000_weight['ID'].str[:2] =='SZ', CSI1000_weight['ID'].str[2:].astype(int) + 2000000, CSI1000_weight['ID'].str[2:].astype(int) + 1000000)
CSI1000_weight = CSI1000_weight.rename(columns={'ID':'skey'})
CSI1000_weight['index_id'] = 1000852
CSI1000_weight['index_name'] = 'CSI1000'
CSI1000_weight = CSI1000_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = CSI1000_weight.groupby('date')['weight'].sum().reset_index()
assert(k[k['weight'] - 100 > 0.02].shape[0] == 0)
write_memb_data(db1, 'index_memb', CSI1000_weight)
print(CSI1000_weight)
CSI1000_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\CSI1000.csv')

weight_table['date'] = weight_table['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
weight_table['ID'] = np.where(weight_table['ID'].str[:2] =='SZ', weight_table['ID'].str[2:].astype(int) + 2000000, weight_table['ID'].str[2:].astype(int) + 1000000)
weight_table = weight_table.rename(columns={'ID':'skey'})
weight_table['index_id'] = 1000985
weight_table['index_name'] = 'CSIRest'
weight_table = weight_table.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = weight_table.groupby('date')['weight'].sum().reset_index()
assert(k[k['weight'] - 100 > 0.02].shape[0] == 0)
write_memb_data(db1, 'index_memb', weight_table)
print(weight_table)
weight_table.to_csv('E:\\daily_index_result\\' + startDate + '\\weight_table.csv')

TSLPy3.Disconnect()