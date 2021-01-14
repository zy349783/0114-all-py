#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


def read_memb_daily(db, name, start_date=None, end_date=None, skey=None, index_id=None, interval=None, col=None,
                    return_sdi=True):
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
            m_ax = pd.DataFrame.from_records(
                collection.find({'index_id': {'$in': [symbol]}}).sort([('date', -1)]).skip(0).limit(1))['date'].values[
                0]
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

TSLPy3.ConnectServer("tsl.tinysoft.com.cn", 443)
dl = TSLPy3.LoginServer("jqtz", "+7.1q2w3e")
assert (dl[0] == 0)

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
    weight_table = pd.DataFrame(TSLPy3.RemoteExecute(tsstr, [], {})[1])
    weight_table.columns = ['date', 'weight', 'ID']
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
sumWeightToday.rename(columns={'weight': 'sumWeightDay'}, inplace=True)
weight_table = CSIRest_weight.merge(sumWeightToday, on='date', how='left')
weight_table['weight'] = weight_table['weight'] / weight_table['sumWeightDay'] * 100
weight_table = weight_table.drop(columns={'sumWeightDay'})

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
IF_weight['ID'] = np.where(IF_weight['ID'].str[:2] == 'SZ', IF_weight['ID'].str[2:].astype(int) + 2000000,
                           IF_weight['ID'].str[2:].astype(int) + 1000000)
IF_weight = IF_weight.rename(columns={'ID': 'skey'})
IF_weight['index_id'] = 1000300
IF_weight['index_name'] = 'IF'
IF_weight = IF_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = IF_weight.groupby('date')['weight'].sum().reset_index()
assert (k[k['weight'] - 100 > 0.02].shape[0] == 0)
#write_memb_data(db1, 'index_memb', IF_weight)
print(IF_weight)
#os.mkdir('E:\\daily_index_result\\' + startDate)
IF_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\IF.csv')

IC_weight['date'] = IC_weight['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
IC_weight['ID'] = np.where(IC_weight['ID'].str[:2] == 'SZ', IC_weight['ID'].str[2:].astype(int) + 2000000,
                           IC_weight['ID'].str[2:].astype(int) + 1000000)
IC_weight = IC_weight.rename(columns={'ID': 'skey'})
IC_weight['index_id'] = 1000905
IC_weight['index_name'] = 'IC'
IC_weight = IC_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = IC_weight.groupby('date')['weight'].sum().reset_index()
assert (k[k['weight'] - 100 > 0.02].shape[0] == 0)
#write_memb_data(db1, 'index_memb', IC_weight)
print(IC_weight)
IC_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\IC.csv')

CSI1000_weight['date'] = CSI1000_weight['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
CSI1000_weight['ID'] = np.where(CSI1000_weight['ID'].str[:2] == 'SZ',
                                CSI1000_weight['ID'].str[2:].astype(int) + 2000000,
                                CSI1000_weight['ID'].str[2:].astype(int) + 1000000)
CSI1000_weight = CSI1000_weight.rename(columns={'ID': 'skey'})
CSI1000_weight['index_id'] = 1000852
CSI1000_weight['index_name'] = 'CSI1000'
CSI1000_weight = CSI1000_weight.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = CSI1000_weight.groupby('date')['weight'].sum().reset_index()
assert (k[k['weight'] - 100 > 0.02].shape[0] == 0)
#write_memb_data(db1, 'index_memb', CSI1000_weight)
print(CSI1000_weight)
CSI1000_weight.to_csv('E:\\daily_index_result\\' + startDate + '\\CSI1000.csv')

weight_table['date'] = weight_table['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
weight_table['ID'] = np.where(weight_table['ID'].str[:2] == 'SZ', weight_table['ID'].str[2:].astype(int) + 2000000,
                              weight_table['ID'].str[2:].astype(int) + 1000000)
weight_table = weight_table.rename(columns={'ID': 'skey'})
weight_table['index_id'] = 1000985
weight_table['index_name'] = 'CSIRest'
weight_table = weight_table.sort_values(by=['date', 'skey']).reset_index(drop=True)
k = weight_table.groupby('date')['weight'].sum().reset_index()
assert (k[k['weight'] - 100 > 0.02].shape[0] == 0)
#(db1, 'index_memb', weight_table)
print(weight_table)
weight_table.to_csv('E:\\daily_index_result\\' + startDate + '\\weight_table.csv')

TSLPy3.Disconnect()

import pymongo
import pandas as pd
import pickle
import datetime
import time
import gzip
import lzma
import pytz


def DB(host, db_name, user, passwd):
    auth_db = db_name if user not in ('admin', 'root') else 'admin'
    uri = 'mongodb://%s:%s@%s/?authSource=%s' % (user, passwd, host, auth_db)
    return DBObj(uri, db_name=db_name)


class DBObj(object):
    def __init__(self, uri, symbol_column='skey', db_name='white_db'):
        self.db_name = db_name
        self.uri = uri
        self.client = pymongo.MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.chunk_size = 20000
        self.symbol_column = symbol_column
        self.date_column = 'date'

    def parse_uri(self, uri):
        # mongodb://user:password@example.com
        return uri.strip().replace('mongodb://', '').strip('/').replace(':', ' ').replace('@', ' ').split(' ')

    def drop_table(self, table_name):
        self.db.drop_collection(table_name)

    def rename_table(self, old_table, new_table):
        self.db[old_table].rename(new_table)

    def write(self, table_name, df):
        if len(df) == 0: return

        multi_date = False

        if self.date_column in df.columns:
            date = str(df.head(1)[self.date_column].iloc[0])
            multi_date = len(df[self.date_column].unique()) > 1
        else:
            raise Exception('DataFrame should contain date column')

        collection = self.db[table_name]
        collection.create_index([('date', pymongo.ASCENDING), ('symbol', pymongo.ASCENDING)], background=True)
        collection.create_index([('symbol', pymongo.ASCENDING), ('date', pymongo.ASCENDING)], background=True)

        if multi_date:
            for (date, symbol), sub_df in df.groupby([self.date_column, self.symbol_column]):
                date = str(date)
                symbol = int(symbol)
                collection.delete_many({'date': date, 'symbol': symbol})
                self.write_single(collection, date, symbol, sub_df)
        else:
            for symbol, sub_df in df.groupby([self.symbol_column]):
                collection.delete_many({'date': date, 'symbol': symbol})
                self.write_single(collection, date, symbol, sub_df)

    def write_single(self, collection, date, symbol, df):
        for start in range(0, len(df), self.chunk_size):
            end = min(start + self.chunk_size, len(df))
            df_seg = df[start:end]
            version = 1
            seg = {'ver': version, 'data': self.ser(df_seg, version), 'date': date, 'symbol': symbol, 'start': start}
            collection.insert_one(seg)

    def build_query(self, start_date=None, end_date=None, symbol=None):
        query = {}

        def parse_date(x):
            if type(x) == str:
                if len(x) != 8:
                    raise Exception("`date` must be YYYYMMDD format")
                return x
            elif type(x) == datetime.datetime or type(x) == datetime.date:
                return x.strftime("%Y%m%d")
            elif type(x) == int:
                return parse_date(str(x))
            else:
                raise Exception("invalid `date` type: " + str(type(x)))

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

        if symbol:
            if type(symbol) == list or type(symbol) == tuple:
                query['symbol'] = {'$in': [parse_symbol(x) for x in symbol]}
            else:
                query['symbol'] = parse_symbol(symbol)

        return query

    def delete(self, table_name, start_date=None, end_date=None, symbol=None):
        collection = self.db[table_name]

        query = self.build_query(start_date, end_date, symbol)
        if not query:
            print('cannot delete the whole table')
            return None

        collection.delete_many(query)

    def read(self, table_name, start_date=None, end_date=None, symbol=None):
        collection = self.db[table_name]

        query = self.build_query(start_date, end_date, symbol)
        if not query:
            print('cannot read the whole table')
            return None

        segs = []
        for x in collection.find(query):
            x['data'] = self.deser(x['data'], x['ver'])
            segs.append(x)
        segs.sort(key=lambda x: (x['symbol'], x['date'], x['start']))
        return pd.concat([x['data'] for x in segs], ignore_index=True) if segs else None

    def list_tables(self):
        return self.db.collection_names()

    def list_dates(self, table_name, start_date=None, end_date=None, symbol=None):
        collection = self.db[table_name]
        dates = set()
        if start_date is None:
            start_date = '00000000'
        if end_date is None:
            end_date = '99999999'
        for x in collection.find(self.build_query(start_date, end_date, symbol), {"date": 1, '_id': 0}):
            dates.add(x['date'])
        return sorted(list(dates))

    def ser(self, s, version):
        pickle_protocol = 4
        if version == 1:
            return gzip.compress(pickle.dumps(s, protocol=pickle_protocol), compresslevel=2)
        elif version == 2:
            return lzma.compress(pickle.dumps(s, protocol=pickle_protocol), preset=1)
        else:
            raise Exception('unknown version')

    def deser(self, s, version):
        def unpickle(s):
            return pickle.loads(s)

        if version == 1:
            return unpickle(gzip.decompress(s))
        elif version == 2:
            return unpickle(lzma.decompress(s))
        else:
            raise Exception('unknown version')


def patch_pandas_pickle():
    if pd.__version__ < '0.24':
        import sys
        from types import ModuleType
        from pandas.core.internals import BlockManager
        pkg_name = 'pandas.core.internals.managers'
        if pkg_name not in sys.modules:
            m = ModuleType(pkg_name)
            m.BlockManager = BlockManager
            sys.modules[pkg_name] = m


patch_pandas_pickle()

import pandas as pd
import random
import numpy as np
import glob
import pickle
import os
import datetime
import time
import TSLPy3

ul = pd.read_csv(r'D:\work\project 17 AMAC\tickStockList_AMAC.csv')
ul = ul['StockID'].values

startDate = datetime.datetime.today().strftime('%Y%m%d') + 'T'
endDate = datetime.datetime.today().strftime('%Y%m%d') + 'T'

import sys

sys.path.append('C:\\Program Files\\Tinysoft\\Analyse.NET')
import TSLPy3

TSLPy3.ConnectServer("tsl.tinysoft.com.cn", 443)
dl = TSLPy3.LoginServer("jqtz", "+7.1q2w3e")
assert (dl[0] == 0)

all_data = []
for num in range(len(ul)):
    stock = ul[num]
    tickname = 'Tick_' + stock
    if num % 10 == 0: print('Processing ' + str(num) + ' AMAC ' + stock)
    tsstr = """
           BegT :=%s;
           EndT :=%s + 0.99;
           setSysParam(pn_stock(),'%s');
           returnData := select ['date'],['close'],['sectional_open'],['sectional_vol'],['sectional_amount']
                         from tradetable datekey BegT to EndT of DefaultStockID() end;
           return returnData;
           """ % (startDate, endDate, stock)
    Tick_Stock = pd.DataFrame(TSLPy3.RemoteExecute(tsstr, {})[1])
    Tick_Stock.columns = list(pd.Series(Tick_Stock.columns).str.decode('GBK'))
    Tick_Stock['intdate'] = Tick_Stock.date.astype(int)
    Tick_Stock['time'] = Tick_Stock.date.map(lambda x: datetime.datetime.utcfromtimestamp(round((x - 25569) * 86400.0)))
    Tick_Stock['adjTime'] = Tick_Stock.date.map(
        lambda x: datetime.datetime.utcfromtimestamp(round((x - 25569) * 86400.0) - 1))
    Tick_Stock['minute'] = Tick_Stock.adjTime.map(lambda x: (x.hour * 60 + x.minute + 1))
    assert (Tick_Stock.minute.max() >= 900) & (Tick_Stock.minute.min() <= 570)
    Tick_Stock['morning'] = np.where(Tick_Stock.minute <= 690, 1, 0)
    Tick_Stock.rename(
        columns={'sectional_open': 'industry_open', 'sectional_vol': 'cum_volume', 'sectional_amount': 'cum_amount'},
        inplace=True)
    Tick_Stock = Tick_Stock[
        ['intdate', 'minute', 'morning', 'time', 'close', 'industry_open', 'cum_volume', 'cum_amount']].reset_index(
        drop=True)
    Tick_Stock['ID'] = stock
    ## ordering per day per stock
    for intD in Tick_Stock.intdate.unique():
        Tick_Stock.loc[Tick_Stock.intdate == intD, 'ordering'] = range(0, len(
            Tick_Stock.loc[Tick_Stock.intdate == intD, 'ID']))
    Tick_Stock['month'] = Tick_Stock.time.dt.month + Tick_Stock.time.dt.year * 100
    test = Tick_Stock

    test['date'] = test['time'].astype(str).apply(lambda x: int(x.split(' ')[0].replace("-", "")))
    test['time'] = test['time'].astype(str).apply(lambda x: int(x.split(' ')[1].replace(":", "")))
    test['datetime'] = test['date'] * 1000000 + test['time']
    test["clockAtArrival"] = test["datetime"].astype(str).apply(
        lambda x: np.int64(datetime.datetime.strptime(x, '%Y%m%d%H%M%S').timestamp() * 1e6))
    test['datetime'] = test["clockAtArrival"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1e6))
    test['time'] = test['time'].astype('int64') * 1000000
    test['skey'] = test['ID'].str[4:].astype(int) + 3000000
    test = test.rename(columns={'industry_open': "open"})
    test['open1'] = test.groupby(['skey', 'date'])['open'].transform('max')
    test = test.sort_values(by=['skey', 'date', 'ordering'])
    test['close1'] = test.groupby(['skey'])['close'].shift(1)
    # 每天只有一条tick cum_volume==0, 且当时open==today's open, close==yesterday's close
    # 20180215, 20180220, 20180221不满足条件，这几天无交易，close=0；20180215, 20180222 close close1也无法对上
    assert (sum(test[test['cum_volume'] == 0].groupby(['skey', 'date'])['ordering'].size() != 1) == 0)
    assert (sum(test[test['cum_volume'] == 0].groupby(['skey', 'date'])['ordering'].unique() != 0) == 0)
    try:
        assert ((test[test['cum_volume'] == 0]['open'].min() > 0) & (test[test['open'] != test['open1']].shape[0] == 0))
        assert (sum(test[(test['cum_volume'] == 0) & (~test['close1'].isnull())]['close'] !=
                    test[(test['cum_volume'] == 0) & (~test['close1'].isnull())]['close1']) == 0)
    except:
        print(test[(test['cum_volume'] == 0) & (test['open'] == 0)]['datetime'].unique())
        print(test[(test['cum_volume'] == 0) & (~test['close1'].isnull())][
                  test[(test['cum_volume'] == 0) & (~test['close1'].isnull())]['close'] !=
                  test[(test['cum_volume'] == 0) & (~test['close1'].isnull())]['close1']]['datetime'].unique())
    test = test[test['cum_volume'] != 0]
    test = test.sort_values(by=['skey', 'date', 'ordering'])
    test = test[["skey", "date", "time", "clockAtArrival", "datetime", "cum_volume", "cum_amount",
                 "open", "close"]]
    # change to second level tick data
    k1 = test.groupby(['date', 'skey'])['datetime'].min().reset_index()
    k1 = k1.rename(columns={'datetime': 'min'})
    k2 = test.groupby(['date', 'skey'])['datetime'].max().reset_index()
    k2 = k2.rename(columns={'datetime': 'max'})
    k = pd.merge(k1, k2, on=['date', 'skey'])
    k['diff'] = (k['max'] - k['min']).apply(lambda x: x.seconds)

    df = pd.DataFrame()
    for i in np.arange(k.shape[0]):
        df1 = pd.DataFrame()
        df1['datetime1'] = [k.loc[i, 'min'] + datetime.timedelta(seconds=int(x)) for x in
                            np.arange(0, k.loc[i, 'diff'] + 1)]
        df1['skey'] = k.loc[i, 'skey']
        df1['date'] = k.loc[i, 'date']
        assert (df1['datetime1'].min() == k.loc[i, 'min'])
        assert (df1['datetime1'].max() == k.loc[i, 'max'])
        df = pd.concat([df, df1])
    test = pd.merge(test, df, left_on=['skey', 'datetime', 'date'], right_on=['skey', 'datetime1', 'date'],
                    how='outer').sort_values(by=['skey', 'date', 'datetime1']).reset_index(drop=True)
    assert (test[test['datetime1'].isnull()].shape[0] == 0)
    for cols in ['cum_volume', 'cum_amount', 'open', 'close']:
        test[cols] = test.groupby(['skey', 'date'])[cols].ffill()
    test.drop(["datetime"], axis=1, inplace=True)
    test = test.rename(columns={'datetime1': 'datetime'})
    test['skey'] = test['skey'].astype('int32')
    test["time"] = test['datetime'].astype(str).apply(lambda x: int(x.split(' ')[1].replace(':', ""))).astype(np.int64)
    test['SendingTime'] = test['date'] * 1000000 + test['time']
    test["clockAtArrival"] = test["SendingTime"].astype(str).apply(
        lambda x: np.int64(datetime.datetime.strptime(x, '%Y%m%d%H%M%S').timestamp() * 1e6))
    test.drop(["SendingTime"], axis=1, inplace=True)
    test['time'] = test['time'] * 1000000

    assert (sum(test[test["open"] != 0].groupby(["skey", 'date'])["open"].nunique() != 1) == 0)
    test["open"] = np.where(test["cum_volume"] > 0, test.groupby(["skey", 'date'])["open"].transform("max"),
                            test["open"])
    assert (sum(test[test["open"] != 0].groupby(["skey", 'date'])["open"].nunique() != 1) == 0)
    assert (test[test["cum_volume"] > 0]["open"].min() > 0)

    test['date'] = test['date'].astype('int32')
    test['cum_volume'] = test['cum_volume'].astype('int64')

    m_in = test[test['time'] <= 113500000000].groupby('skey').last()['time'].min()
    m_ax = test[test['time'] >= 125500000000].groupby('skey').first()['time'].max()
    assert (test[(test['time'] >= m_in) & (test['time'] <= m_ax)].drop_duplicates(['cum_volume', 'open',
                                                                                   'close', 'cum_amount', 'skey',
                                                                                   'date'], keep=False).shape[0] == 0
            & (sum(test[(test['time'] >= m_in) & (test['time'] <= m_ax)].groupby('skey')[
                       'cum_volume'].nunique() != 1) == 0) &
            (sum(test[(test['time'] >= m_in) & (test['time'] <= m_ax)].groupby('skey')['close'].nunique() != 1) == 0))
    test = pd.concat([test[test['time'] <= 113500000000], test[test['time'] >= 125500000000]])

    test = test.sort_values(by=["skey", 'date', 'time'])
    test["ordering"] = test.groupby(["skey", 'date']).cumcount() + 1
    test['ordering'] = test['ordering'].astype('int32')

    for cols in ['open', 'cum_amount', 'close']:
        test[cols] = test[cols].apply(lambda x: round(x, 4)).astype('float64')
    assert (test['time'].max() < 150500000000)

    test = test[["skey", "date", "time", "clockAtArrival", "datetime", "ordering", "cum_volume", "cum_amount",
                 "open", "close"]]

    print("index finished")

    database_name = 'com_md_eq_cn'
    user = "zhenyuy"
    password = "bnONBrzSMGoE"

    db1 = DB("192.168.10.178", database_name, user, password)
    # db1.write('md_index', test)
    all_data = all_data + [test]

    del test

all_data = pd.concat(all_data).reset_index(drop=True)
all_data.to_csv('E:\\AMAC_index\\' + datetime.datetime.today().strftime('%Y%m%d') + '.csv')

import pymongo
import pandas as pd
import numpy as np
import pickle
import datetime
import time
import gzip
import lzma
import pytz


def DB(host, db_name, user, passwd):
    auth_db = db_name if user not in ('admin', 'root') else 'admin'
    url = 'mongodb://%s:%s@%s/?authSource=%s' % (user, passwd, host, auth_db)
    client = pymongo.MongoClient(url, maxPoolSize=None)
    db = client[db_name]
    return db


def read_memb_daily(db, name, start_date=None, end_date=None, skey=None, index_id=None, interval=None, col=None,
                    return_sdi=True):
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
            m_ax = pd.DataFrame.from_records(
                collection.find({'index_id': {'$in': [symbol]}}).sort([('date', -1)]).skip(0).limit(1))['date'].values[
                0]
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


def read_stock_daily(db, name, start_date=None, end_date=None, skey=None, index_name=None, interval=None, col=None,
                     return_sdi=True):
    collection = db[name]
    # Build projection
    prj = {'_id': 0}
    if col is not None:
        if return_sdi:
            col = ['skey', 'date'] + col
        for col_name in col:
            prj[col_name] = 1

    # Build query
    query = {}
    if skey is not None:
        query['skey'] = {'$in': skey}
    if index_name is not None:
        query['index_name'] = {'$in': index_name}
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
        df = df.sort_values(by=['date', 'skey'])
    return df


database_name = 'com_md_eq_cn'
user = "zhenyuy"
password = "bnONBrzSMGoE"

pd.set_option('max_columns', 200)
db1 = DB("192.168.10.178", database_name, user, password)

from WindPy import *

w.start()

import os
import glob
import time
import datetime
import pandas as pd

pd.set_option('display.max_columns', 200)
pd.options.mode.chained_assignment = None
import numpy as np

database_name = 'com_md_eq_cn'
user = "zhenyuy"
password = "bnONBrzSMGoE"

il = pd.read_csv(r'D:\work\project 17 AMAC\tickStockList_AMAC.csv')
il['StockID'] = il['StockID'].str[3:] + '.CSI'
il = il['StockID'].values

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

import sys

sys.path.append('C:\\Program Files\\Tinysoft\\Analyse.NET')
import TSLPy3

TSLPy3.ConnectServer("tsl.tinysoft.com.cn", 443)
dl = TSLPy3.LoginServer("jqtz", "+7.1q2w3e")
assert (dl[0] == 0)


def updateAShare(date):
    TRDate = str(date)
    tsstr = """
           BegT:=%s;
           EndT:=%s;
           SetSysParam(pn_stock(),'SH000001');
           SetSysParam(PN_Cycle(),cy_day());
           dateArr:=MarketTradeDayQk(BegT,EndT);
           r:=array();
           for nI:=0 to length(dateArr)-1 do
           begin
             echo dateArr[nI];
             t:= getabkbydate('A股',dateArr[nI]);
             r:=r union2 t;
           end;
           r:= select [0] as 'StockID' from `r end;
           r := select * from r order by ['StockID'] end;
           return r;
            """ % (TRDate + 'T', TRDate + 'T + 0.99')
    stockList = pd.DataFrame(TSLPy3.RemoteExecute(tsstr, [], {})[1])
    stockList.columns = list(pd.Series(stockList.columns).str.decode('GBK'))
    stockList['StockID'] = stockList['StockID'].str.decode('GBK')
    stockList['skey'] = np.where(stockList['StockID'].str[:2] == 'SH',
                                 1000000 + stockList['StockID'].str[2:].astype(int),
                                 2000000 + stockList['StockID'].str[2:].astype(int))
    stockList['date'] = int(TRDate)
    return stockList


dl = [int(datetime.datetime.today().strftime('%Y%m%d'))]
total_stock = []
for d in dl:
    data = updateAShare(d)
    total_stock += [data]
total_stock = pd.concat(total_stock, sort=False)

try:
    test = w.wset("indexconstituent", "date=%s; windcode=%s" % (d, il[0]))
    assert (int(test.Data[0][0].strftime('%Y%m%d')) == d)

    data2 = []
    add = []
    save = []
    startTm = datetime.datetime.now()
    for d in dl:
        data1 = []
        for i in il:
            data = w.wset("indexconstituent", "date=%s; windcode=%s" % (d, i))
            df = pd.DataFrame(data=np.array(data.Data).T, columns=data.Fields)
            df['index_id'] = 3000000 + int(i[1:6])
            try:
                assert (abs(df['i_weight'].sum() - 100) < 0.1)
            except:
                print('sum of weight far from 100')
                print(df['i_weight'].sum())
                print(df)
                save += [df]
            df['skey'] = df['wind_code'].str[:-3].astype(int)
            df['skey'] = np.where(df['skey'] < 600000, df['skey'] + 2000000, df['skey'] + 1000000)
            df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
            corrections = {3011030: 'AMAC 农林',
                           3011031: 'AMAC 采矿',
                           3011041: 'AMAC 公用',
                           3011042: 'AMAC 建筑',
                           3011043: 'AMAC 交运',
                           3011044: 'AMAC 信息',
                           3011045: 'AMAC 批零',
                           3011046: 'AMAC 金融',
                           3011047: 'AMAC 地产',
                           3011049: 'AMAC 文体',
                           3011050: 'AMAC 综企',
                           3030036: 'AMAC 餐饮',
                           3030037: 'AMAC 商务',
                           3030038: 'AMAC 科技',
                           3030039: 'AMAC 公共',
                           3030040: 'AMAC 社会',
                           3030041: 'AMAC 农副',
                           3030042: 'AMAC 食品',
                           3030043: 'AMAC 饮料',
                           3030044: 'AMAC 纺织',
                           3030045: 'AMAC 服装',
                           3030046: 'AMAC 皮革',
                           3030047: 'AMAC 木材',
                           3030048: 'AMAC 家具',
                           3030049: 'AMAC 造纸',
                           3030050: 'AMAC 印刷',
                           3030051: 'AMAC 文教',
                           3030052: 'AMAC 石化',
                           3030053: 'AMAC 化学',
                           3030054: 'AMAC 医药',
                           3030055: 'AMAC 化纤',
                           3030056: 'AMAC 橡胶',
                           3030057: 'AMAC 矿物',
                           3030058: 'AMAC 钢铁',
                           3030059: 'AMAC 有色',
                           3030060: 'AMAC 金属',
                           3030061: 'AMAC 通用',
                           3030062: 'AMAC 专用',
                           3030063: 'AMAC 汽车',
                           3030064: 'AMAC 运输',
                           3030065: 'AMAC 电气',
                           3030066: 'AMAC 电子',
                           3030067: 'AMAC 仪表'}
            df['index_name'] = df['index_id']
            df.index_name = df.index_name.map(corrections)
            data1 += [df]
        data1 = pd.concat(data1).reset_index(drop=True)
        stock_list = list(set(total_stock[total_stock['date'] == d]['skey'].unique()) - set(data1['skey'].unique()))
        stock_list = [str(i - 1000000).rjust(6, "0") + '.SH' if i < 2000000 else str(i - 2000000).rjust(6, "0") + '.SZ'
                      for i in stock_list]
        dd = str(d)[:4] + '-' + str(d)[4:6] + '-' + str(d)[6:]
        add1 = pd.DataFrame(columns=['date', 'stock_list'])
        add1['stock_list'] = stock_list
        add1['date'] = dd
        add += [add1]
        data1 = data1.rename(columns={'i_weight': 'weight'})
        data1 = data1[['date', 'skey', 'index_id', 'index_name', 'weight']]
        data2 += [data1]
    data2 = pd.concat(data2).reset_index(drop=True)
    data2 = data2.drop_duplicates(keep='first')
    add = pd.concat(add).reset_index(drop=True)
    print('get index composition weight')
    print(datetime.datetime.now() - startTm)

    data3 = []
    startTm = datetime.datetime.now()
    stock_list = add['stock_list'].unique()
    for s in stock_list:
        start_date = add[add['stock_list'] == s]['date'].min()
        end_date = add[add['stock_list'] == s]['date'].max()
        add_data = w.wsd(s, "industry_CSRCcode12", start_date, end_date, "industryType=3;PriceAdj=F")
        if add_data.ErrorCode != 0:
            continue
        nd = pd.DataFrame(data=np.array(add_data.Data).T, columns=['Ind'])
        nd1 = pd.DataFrame(data=np.array(add_data.Times).T, columns=['DateTime'])
        nd = pd.concat([nd1, nd], axis=1)
        nd = nd[~nd['Ind'].isnull()]
        if nd.empty:
            continue
        else:
            nd['index_id'] = nd['Ind'].str[1:].astype(int)
            nd['date'] = nd['DateTime'].astype(str).apply(lambda x: x.replace('-', '')).astype(int)
            if s[-2:] == 'SZ':
                nd['skey'] = int(s[:-3]) + 2000000
            else:
                nd['skey'] = int(s[:-3]) + 1000000
            nd['weight'] = 0
            data3 += [nd]
    data3 = pd.concat(data3).reset_index(drop=True)
    print(datetime.datetime.now() - startTm)
    print('get extra data')

    data2 = pd.concat([data2, data3])
    data2 = data2.sort_values(by=['date', 'skey', 'weight']).reset_index(drop=True)
    if data2[data2.duplicated(['date', 'skey'], keep=False)].shape[0] != 0:
        display(data2[data2.duplicated(['date', 'skey'], keep=False)])
        data2 = data2.drop_duplicates(['date', 'skey'], keep='last').reset_index(drop=True)
    assert (data2[data2['index_id'] < 100]['weight'].unique() == [0])

    data2['index_id'] = np.where(data2.index_id <= 5, 3011030, np.where(data2.index_id <= 12, 3011031,
                                                                        np.where(data2.index_id == 13, 3030041,
                                                                                 np.where(data2.index_id == 14, 3030042,
                                                                                          np.where(data2.index_id == 15,
                                                                                                   3030043, np.where(
                                                                                                  data2.index_id == 17,
                                                                                                  3030044, np.where(
                                                                                                      data2.index_id == 18,
                                                                                                      3030045, np.where(
                                                                                                          data2.index_id == 19,
                                                                                                          3030046,
                                                                                                          np.where(
                                                                                                              data2.index_id == 20,
                                                                                                              3030047,
                                                                                                              np.where(
                                                                                                                  data2.index_id == 21,
                                                                                                                  3030048,
                                                                                                                  np.where(
                                                                                                                      data2.index_id == 22,
                                                                                                                      3030049,
                                                                                                                      np.where(
                                                                                                                          data2.index_id == 23,
                                                                                                                          3030050,
                                                                                                                          np.where(
                                                                                                                              data2.index_id == 24,
                                                                                                                              3030051,
                                                                                                                              np.where(
                                                                                                                                  data2.index_id == 25,
                                                                                                                                  3030052,
                                                                                                                                  np.where(
                                                                                                                                      data2.index_id == 26,
                                                                                                                                      3030053,
                                                                                                                                      np.where(
                                                                                                                                          data2.index_id == 27,
                                                                                                                                          3030054,
                                                                                                                                          np.where(
                                                                                                                                              data2.index_id == 28,
                                                                                                                                              3030055,
                                                                                                                                              np.where(
                                                                                                                                                  data2.index_id == 29,
                                                                                                                                                  3030056,
                                                                                                                                                  np.where(
                                                                                                                                                      data2.index_id == 30,
                                                                                                                                                      3030057,
                                                                                                                                                      np.where(
                                                                                                                                                          data2.index_id == 31,
                                                                                                                                                          3030058,
                                                                                                                                                          np.where(
                                                                                                                                                              data2.index_id == 32,
                                                                                                                                                              3030059,
                                                                                                                                                              np.where(
                                                                                                                                                                  data2.index_id == 33,
                                                                                                                                                                  3030060,
                                                                                                                                                                  np.where(
                                                                                                                                                                      data2.index_id == 34,
                                                                                                                                                                      3030061,
                                                                                                                                                                      np.where(
                                                                                                                                                                          data2.index_id == 35,
                                                                                                                                                                          3030062,
                                                                                                                                                                          np.where(
                                                                                                                                                                              data2.index_id == 36,
                                                                                                                                                                              3030063,
                                                                                                                                                                              np.where(
                                                                                                                                                                                  data2.index_id == 37,
                                                                                                                                                                                  3030064,
                                                                                                                                                                                  np.where(
                                                                                                                                                                                      data2.index_id == 38,
                                                                                                                                                                                      3030065,
                                                                                                                                                                                      np.where(
                                                                                                                                                                                          data2.index_id == 39,
                                                                                                                                                                                          3030066,
                                                                                                                                                                                          np.where(
                                                                                                                                                                                              data2.index_id == 40,
                                                                                                                                                                                              3030067,
                                                                                                                                                                                              np.where(
                                                                                                                                                                                                  data2.index_id <= 43,
                                                                                                                                                                                                  3011050,
                                                                                                                                                                                                  np.where(
                                                                                                                                                                                                      data2.index_id <= 46,
                                                                                                                                                                                                      3011041,
                                                                                                                                                                                                      np.where(
                                                                                                                                                                                                          data2.index_id <= 50,
                                                                                                                                                                                                          3011042,
                                                                                                                                                                                                          np.where(
                                                                                                                                                                                                              data2.index_id <= 52,
                                                                                                                                                                                                              3011045,
                                                                                                                                                                                                              np.where(
                                                                                                                                                                                                                  data2.index_id <= 60,
                                                                                                                                                                                                                  3011043,
                                                                                                                                                                                                                  np.where(
                                                                                                                                                                                                                      data2.index_id <= 62,
                                                                                                                                                                                                                      3030036,
                                                                                                                                                                                                                      np.where(
                                                                                                                                                                                                                          data2.index_id <= 65,
                                                                                                                                                                                                                          3011044,
                                                                                                                                                                                                                          np.where(
                                                                                                                                                                                                                              data2.index_id <= 69,
                                                                                                                                                                                                                              3011046,
                                                                                                                                                                                                                              np.where(
                                                                                                                                                                                                                                  data2.index_id == 70,
                                                                                                                                                                                                                                  3011047,
                                                                                                                                                                                                                                  np.where(
                                                                                                                                                                                                                                      data2.index_id <= 72,
                                                                                                                                                                                                                                      3030037,
                                                                                                                                                                                                                                      np.where(
                                                                                                                                                                                                                                          data2.index_id <= 75,
                                                                                                                                                                                                                                          3030038,
                                                                                                                                                                                                                                          np.where(
                                                                                                                                                                                                                                              data2.index_id <= 78,
                                                                                                                                                                                                                                              3030039,
                                                                                                                                                                                                                                              np.where(
                                                                                                                                                                                                                                                  data2.index_id <= 81,
                                                                                                                                                                                                                                                  3030040,
                                                                                                                                                                                                                                                  np.where(
                                                                                                                                                                                                                                                      data2.index_id == 82,
                                                                                                                                                                                                                                                      3011049,
                                                                                                                                                                                                                                                      np.where(
                                                                                                                                                                                                                                                          data2.index_id <= 84,
                                                                                                                                                                                                                                                          3030040,
                                                                                                                                                                                                                                                          np.where(
                                                                                                                                                                                                                                                              data2.index_id <= 89,
                                                                                                                                                                                                                                                              3011049,
                                                                                                                                                                                                                                                              np.where(
                                                                                                                                                                                                                                                                  data2.index_id == 90,
                                                                                                                                                                                                                                                                  3011050,
                                                                                                                                                                                                                                                                  data2[
                                                                                                                                                                                                                                                                      'index_id']))))))))))))))))))))))))))))))))))))))))))))))
    assert (data2['index_id'].min() > 100)
    corrections = {3011030: 'AMAC 农林',
                   3011031: 'AMAC 采矿',
                   3011041: 'AMAC 公用',
                   3011042: 'AMAC 建筑',
                   3011043: 'AMAC 交运',
                   3011044: 'AMAC 信息',
                   3011045: 'AMAC 批零',
                   3011046: 'AMAC 金融',
                   3011047: 'AMAC 地产',
                   3011049: 'AMAC 文体',
                   3011050: 'AMAC 综企',
                   3030036: 'AMAC 餐饮',
                   3030037: 'AMAC 商务',
                   3030038: 'AMAC 科技',
                   3030039: 'AMAC 公共',
                   3030040: 'AMAC 社会',
                   3030041: 'AMAC 农副',
                   3030042: 'AMAC 食品',
                   3030043: 'AMAC 饮料',
                   3030044: 'AMAC 纺织',
                   3030045: 'AMAC 服装',
                   3030046: 'AMAC 皮革',
                   3030047: 'AMAC 木材',
                   3030048: 'AMAC 家具',
                   3030049: 'AMAC 造纸',
                   3030050: 'AMAC 印刷',
                   3030051: 'AMAC 文教',
                   3030052: 'AMAC 石化',
                   3030053: 'AMAC 化学',
                   3030054: 'AMAC 医药',
                   3030055: 'AMAC 化纤',
                   3030056: 'AMAC 橡胶',
                   3030057: 'AMAC 矿物',
                   3030058: 'AMAC 钢铁',
                   3030059: 'AMAC 有色',
                   3030060: 'AMAC 金属',
                   3030061: 'AMAC 通用',
                   3030062: 'AMAC 专用',
                   3030063: 'AMAC 汽车',
                   3030064: 'AMAC 运输',
                   3030065: 'AMAC 电气',
                   3030066: 'AMAC 电子',
                   3030067: 'AMAC 仪表'}
    data2['index_name'] = data2['index_id']
    data2.index_name = data2.index_name.map(corrections)
    data2 = data2[['date', 'skey', 'index_id', 'index_name', 'weight']]
    try:
        assert (abs(data2.groupby(['date', 'index_id'])['weight'].sum() - 100).max() < 0.1)
    except:
        print(data2.groupby(['date', 'index_id'])['weight'].sum())
    write_memb_data(db1, 'index_memb', data2)
    data2.to_csv('E:\\AMAC_weight\\' + str(dl[0]) + '_check.csv')

except:
    print('Wind data not ready')
