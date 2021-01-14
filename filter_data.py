import pymongo
import pandas as pd
import pickle
import datetime
import time
import gzip
import lzma
import pytz
import numpy as np


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

    def read_daily(self, table_name, start_date=None, end_date=None, index_id=None, skey=None, interval=None, col=None,
                   return_sdi=True):
        collection = self.db[table_name]
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


def sta_sizeFilter(stockID, startDate, endDate, regWindowSize=20, weekInterval=1):
    database_name = 'com_md_eq_cn'
    user = "zhenyuy"
    password = "bnONBrzSMGoE"

    pd.set_option('max_columns', 200)
    db = DB("192.168.10.178", database_name, user, password)

    print(' ...... Now Calculating SizeFilter for  ', stockID)
    #    startTm = datetime.datetime.now()
    stockData = db.read('md_snapshot_l2', start_date=startDate, end_date=endDate, symbol=[stockID])
    stockData = stockData.loc[((stockData.bid1p != 0) | (stockData.ask1p != 0)), \
                              ['skey', 'date', 'time', 'clockAtArrival', 'datetime', 'ordering',
                               'cum_amount', 'bid1p', 'bid1q', 'bid5q', 'ask5q']].reset_index(drop=True)
    stockData = stockData[((stockData.time >= 93000000000) & (stockData.time <= 113000000000)) | \
                          ((stockData.time >= 130000000000) & (stockData.time <= 150000000000))].reset_index(drop=True)
    indexDaily = db.read_daily('index_memb', start_date=startDate, end_date=endDate, index_id=[1000905])
    indexDaily['tradeConsDay'] = indexDaily.groupby(['date']).grouper.group_info[0]
    indexDaily = indexDaily.groupby('date')['tradeConsDay'].first().reset_index()
    df_train = stockData.merge(indexDaily[['date', 'tradeConsDay']], how='left', on=['date'], validate='many_to_one')

    df_train = df_train[(df_train['time'] >= 93000000000) & (df_train['time'] < 145655000000)].reset_index(drop=True)
    groupAllData = df_train.groupby(['skey', 'date'])
    df_train['amountThisUpdate'] = df_train.cum_amount - groupAllData['cum_amount'].shift(1)
    df_train['amountThisUpdate'] = np.where(pd.isnull(df_train.amountThisUpdate), df_train.cum_amount,
                                            df_train.amountThisUpdate)

    ### add useful day indicator
    df_train['curNearLimit'] = np.where((df_train.ask5q == 0) | (df_train.bid5q == 0), 1.0, 0.0)
    df_train['curNearLimit_L1'] = groupAllData['curNearLimit'].shift(1)
    df_train['dailyCount'] = groupAllData['time'].transform('count')
    df_train['nearLimitCount'] = groupAllData['curNearLimit'].transform('sum')
    dateInfo = groupAllData['dailyCount', 'nearLimitCount', 'tradeConsDay'].mean().reset_index()
    del groupAllData
    dateInfo['useFlag'] = np.where(dateInfo['nearLimitCount'] * 2 < dateInfo['dailyCount'], 1, 0)
    dateInfo['useConsDay'] = dateInfo['useFlag'].cumsum()
    df_train = pd.merge(df_train, dateInfo[['date', 'tradeConsDay', 'useFlag', 'useConsDay']],
                        how='left', on=['date', 'tradeConsDay'], validate='many_to_one')

    df_train['weekday'] = df_train['datetime'].dt.weekday
    sizeFilterData = df_train.groupby(['date'])['tradeConsDay'].first().reset_index()
    sizeFilterData['amountFilter'] = np.nan
    ## we only update on Thrusday
    regDays = sorted(list(df_train.loc[df_train.weekday == 3, 'tradeConsDay'].unique()))

    weekInterval = 1
    for d in range(int(regWindowSize / 5), len(regDays), weekInterval):
        amountFilter = np.nan
        ## get current Thrusday
        endTradeConsDay = regDays[d]
        endUseConsDay = dateInfo[dateInfo['tradeConsDay'] == endTradeConsDay]['useConsDay'].values[0]
        startUseConsDay = max(endUseConsDay - regWindowSize + 1, 1)

        ## check 60 consecutive days
        if dateInfo['useConsDay'].max() < 1:
            amountFilter = np.nan
            continue
        startTradeConsDay = dateInfo[dateInfo['useConsDay'] == startUseConsDay]['tradeConsDay'].values[0]
        endTradeConsDay = dateInfo[dateInfo['tradeConsDay'] == endTradeConsDay]['tradeConsDay'].values[0]
        if (endTradeConsDay - startTradeConsDay > 59) or (endUseConsDay - startUseConsDay < 9):
            amountFilter = np.nan
            continue
            ## get the Monday right after current Thursday update
        oss_intdate = df_train.loc[df_train.tradeConsDay == endTradeConsDay, 'date'].unique()[0]
        oss_intdate = (datetime.datetime.strptime(str(oss_intdate), '%Y%m%d') - datetime.datetime(1899, 12,
                                                                                                  30)).days + 4
        oss = int((datetime.datetime(1899, 12, 30) + datetime.timedelta(int(oss_intdate))).strftime('%Y%m%d'))
        ## get the Friday right after next Thursday update
        if d >= len(regDays) - weekInterval:
            ose = df_train.date.max()
        else:
            ose_intdate = df_train.loc[df_train.tradeConsDay == regDays[d + weekInterval], 'date'].unique()[0]
            ose_intdate = (datetime.datetime.strptime(str(ose_intdate), '%Y%m%d') - datetime.datetime(1899, 12,
                                                                                                      30)).days + 1
            ose = int((datetime.datetime(1899, 12, 30) + datetime.timedelta(int(ose_intdate))).strftime('%Y%m%d'))
        inSampleSlice = df_train[(df_train.useConsDay >= startUseConsDay) & \
                                 (df_train.useConsDay <= endUseConsDay) & \
                                 (df_train.useFlag == 1)].reset_index(drop=True)
        amountFilter = inSampleSlice[(inSampleSlice['curNearLimit'] == 0) & \
                                     (inSampleSlice['curNearLimit_L1'] == 0)].amountThisUpdate.quantile(.75)
        if ose < oss:
            print('out of sample end day < start day, skip')
            continue
        sizeFilterData.loc[(sizeFilterData.date >= oss) & (sizeFilterData.date <= ose), 'amountFilter'] = amountFilter
    sizeFilterData['skey'] = int(stockID)
    sizeFilterData = sizeFilterData[['skey', 'date', 'amountFilter']]

    #    sizeFilterData.reset_index(drop = True).to_pickle(os.path.join(r'test_path',str(stockID) +'.pkl'))
    #    print(datetime.datetime.now() - startTm)
    #    return sizeFilterData, [stock, df_train.date.max(), amountFilter]
    return sizeFilterData


sta_sizeFilter(2000001, 20170101, 20200924)