import os
import glob
import datetime
import numpy as np
import pandas as pd
import pickle

date = 20200102
readPath = r'\\192.168.10.30\Kevin_zhenyu\rawData\logs_%s_zs_92_01_day_data'%date
mdDataSHPath = glob.glob(os.path.join(readPath, 'mdLog_SH***.csv'))[-1]
mdDataSH = pd.read_csv(mdDataSHPath)
mdindex = mdDataSH[mdDataSH['StockID'] < 600000]
mdindex['time'] = mdindex.time.str.slice(0, 2) + mdindex.time.str.slice(3, 5) + mdindex.time.str.slice(6, 8) + '000'
mdindex['time'] = mdindex['time'].astype('int64')
mdindex['time'] = mdindex.groupby(['StockID'])['time'].cummax()
mdindex['max_cum_volume'] = mdindex.groupby(['StockID'])['cum_volume'].cummax()
mdindex = mdindex[(mdindex['cum_volume'] > 0) & (mdindex['time'] >= 93000000) &\
                    (mdindex['cum_volume'] == mdindex['max_cum_volume'])]
mdindex = mdindex[['StockID', 'clockAtArrival', 'sequenceNo', 'time', 'close']]


mdDataSH = mdDataSH[mdDataSH['StockID'] >= 600000]
mdDataSH['ID'] = mdDataSH['StockID'] + 1000000
mdDataSH['time'] = mdDataSH.time.str.slice(0, 2) + mdDataSH.time.str.slice(3, 5) + mdDataSH.time.str.slice(6, 8) + '000'
mdDataSH['time'] = mdDataSH['time'].astype('int64')
mdDataSH['time'] = mdDataSH.groupby(['ID'])['time'].cummax()
mdDataSH['max_cum_volume'] = mdDataSH.groupby(['StockID'])['cum_volume'].cummax()
mdDataSH = mdDataSH[(mdDataSH['cum_volume'] > 0) & (mdDataSH['time'] >= 93000000) &\
                    (mdDataSH['cum_volume'] == mdDataSH['max_cum_volume'])]
mdDataSH = mdDataSH[['ID', 'clockAtArrival', 'sequenceNo', 'time', 'cum_volume', 'bid1p', 'ask1p', 'bid1q', 'ask1q', 'bid5q', 'ask5q']]


mdDataSZPath = glob.glob(os.path.join(readPath, 'mdLog_SZ***.csv'))[-1]
mdDataSZ = pd.read_csv(mdDataSZPath)
mdDataSZ['ID'] = mdDataSZ['StockID'] + 2000000
mdDataSZ['time'] = mdDataSZ.time.str.slice(0, 2) + mdDataSZ.time.str.slice(3, 5) + mdDataSZ.time.str.slice(6, 8) + '000'
mdDataSZ['time'] = mdDataSZ['time'].astype('int64')
mdDataSZ['time'] = mdDataSZ.groupby(['ID'])['time'].cummax()
mdDataSZ['max_cum_volume'] = mdDataSZ.groupby(['StockID'])['cum_volume'].cummax()
mdDataSZ = mdDataSZ[(mdDataSZ['cum_volume'] > 0) & (mdDataSZ['time'] >= 93000000) &\
                    (mdDataSZ['cum_volume'] == mdDataSZ['max_cum_volume'])]
mdDataSZ = mdDataSZ[['ID', 'clockAtArrival', 'sequenceNo', 'time', 'cum_volume', 'bid1p', 'ask1p', 'bid1q', 'ask1q', 'bid5q', 'ask5q']]

mdData = pd.concat([mdDataSH, mdDataSZ]).reset_index(drop=True)
mdData = mdData.sort_values(by=['ID', 'sequenceNo']).reset_index(drop=True)

mdData['safeBid1p'] = np.where(mdData['bid1p'] == 0, mdData['ask1p'], mdData['bid1p'])
mdData['safeAsk1p'] = np.where(mdData['ask1p'] == 0, mdData['bid1p'], mdData['ask1p'])
mdData['adjMid'] = (mdData['safeBid1p']*mdData['ask1q'] + mdData['safeAsk1p']*mdData['bid1q'])/(mdData['bid1q'] + mdData['ask1q'])

mdData['session'] = np.where(mdData['time'] >= 130000000, 1, 0)
def findTmValue(clockLs, tm, method='L', buffer=0):
    maxIx = len(clockLs)
    orignIx = np.arange(maxIx)
    if method == 'F':
        ix = np.searchsorted(clockLs, clockLs+(tm-buffer))
        ## if target future index is next tick, mask
        mask = (orignIx == (ix - 1))|(orignIx == ix)|(ix == maxIx)
    elif method == 'L':
        ## if target future index is last tick, mask
        ix = np.searchsorted(clockLs, clockLs-(tm-buffer))
        ix = ix - 1
        ix[ix<0] = 0
        mask = (orignIx == ix) | ((clockLs-(tm-buffer)).values < clockLs.values[0])
    ix[mask] = -1
    return ix

mdData = mdData.reset_index(drop=True)
groupAllData = mdData.groupby(['ID', 'session'])
mdData['sessionStartCLA'] = groupAllData['clockAtArrival'].transform('min')
mdData['relativeClock'] = mdData['clockAtArrival'] - mdData['sessionStartCLA']
mdData['trainFlag'] = np.where(mdData['relativeClock'] > 179.5*1e6, 1, 0)
mdData['index'] = mdData.index.values
mdData['sessionStartIx'] = groupAllData['index'].transform('min')
for tm in [90]:
    tmCol = 'F{}s_ix'.format(tm)
    mdData[tmCol] = groupAllData['relativeClock'].transform(lambda x: findTmValue(x, tm*1e6, 'F', 5*1e5)).astype(int)
for tm in [90]:
    tmIx = mdData['F{}s_ix'.format(tm)].values + mdData['sessionStartIx'].values
    adjMid_tm = mdData['adjMid'].values[tmIx]
    adjMid_tm[mdData['F{}s_ix'.format(tm)].values == -1] = np.nan
    mdData['adjMid_F{}s'.format(tm)] = adjMid_tm

mdindex = mdindex.sort_values(by=['StockID', 'sequenceNo']).reset_index(drop=True)
mdindex['session'] = np.where(mdindex['time'] >= 130000000, 1, 0)
mdindex = mdindex.reset_index(drop=True)
groupAllData = mdindex.groupby(['StockID', 'session'])
mdindex['sessionStartCLA'] = groupAllData['clockAtArrival'].transform('min')
mdindex['relativeClock'] = mdindex['clockAtArrival'] - mdindex['sessionStartCLA']
mdindex['trainFlag'] = np.where(mdindex['relativeClock'] > 179.5*1e6, 1, 0)
mdindex['index'] = mdindex.index.values
mdindex['sessionStartIx'] = groupAllData['index'].transform('min')
for tm in [90]:
    tmCol = 'F{}s_ix'.format(tm)
    mdindex[tmCol] = groupAllData['relativeClock'].transform(lambda x: findTmValue(x, tm*1e6, 'F', 5*1e5)).astype(int)
for tm in [90]:
    tmIx = mdindex['F{}s_ix'.format(tm)].values + mdindex['sessionStartIx'].values
    adjMid_tm = mdindex['close'].values[tmIx]
    adjMid_tm[mdindex['F{}s_ix'.format(tm)].values == -1] = np.nan
    mdindex['close_F{}s'.format(tm)] = adjMid_tm
mdindex["indexRet"] = mdindex["close_F90s"] / mdindex["close"] - 1
mdindex = mdindex[mdindex["StockID"] != 16]
mdindex["index1"] = np.where(mdindex["StockID"] == 300, "IF", np.where(
    mdindex["StockID"] == 905, "IC", "CSI1000"
))


readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\equityTradeLogs'
orderLog = pd.read_csv(os.path.join(readPath, 'speedCompare_%s.csv'%date))
orderLog['order'] = orderLog.groupby(['date', 'accCode', 'secid', 'vai']).grouper.group_info[0]
orderLog['orderNtl'] = orderLog['absOrderSize'] * orderLog['orderPrice']
order = orderLog[orderLog["updateType"] == 0]
trade = orderLog[orderLog["updateType"] == 4]
trade["tradeNtl"] = trade["absOrderSizeCumFilled"] * trade["tradePrice"]
order = pd.merge(order, trade.loc[:, ['order', 'clockAtArrival', 'tradePrice', 'tradeNtl']], on='order')
order["interval"] = order["clockAtArrival_y"] - order["clockAtArrival_x"]
order = pd.merge(order, mdData.loc[:, ['ID', 'clockAtArrival', 'cum_volume']], left_on=['secid', 'vai'], right_on=['ID','cum_volume'])
order["newcaa"] = order["interval"] + order["clockAtArrival"]
order = order.loc[:, ["secid", "vai", "orderDirection", "tradePrice_y", "newcaa", "tradeNtl"]]
order = order.rename(columns={"secid": "ID", "tradePrice_y": "tradePrice", "newcaa": "clockAtArrival"})
re = pd.concat([order, mdData.loc[:, ["ID", "clockAtArrival", "cum_volume", "adjMid", "adjMid_F90s"]]])
re = re.sort_values(by=["ID", "clockAtArrival"])
re = re.reset_index(drop=True)
re["adjMid_F90s"] = re.groupby("ID")["adjMid_F90s"].bfill()
re["buyRet"] = np.where(re["orderDirection"] == 1, re["adjMid_F90s"]/re["tradePrice"] - 1, np.nan)
re["sellRet"] = np.where(re["orderDirection"] == -1, re["tradePrice"]/re["adjMid_F90s"] - 1, np.nan)
li_st = pd.read_csv(r'\\192.168.10.30\Kevin_zhenyu\orderLog\code\indexCatInfo.csv', encoding="utf-8")
re = pd.merge(re, li_st, left_on="ID", right_on="secid", how="inner")
re['index1'] = np.where(re["index"] == 'CSIRest', 'CSI1000', re["index"])

re = pd.concat([re, mdindex.loc[:,["index1", "clockAtArrival", "indexRet"]]]).sort_values(by=["clockAtArrival"]).reset_index(drop="True")
re["indexRet"] = re.groupby("index1")["indexRet"].bfill()
re1 = re[~re["vai"].isnull()]
re1 = re1.sort_values(by=["ID", "clockAtArrival"]).reset_index(drop="True")

F1 = open(r"\\192.168.10.30\Kevin_zhenyu\orderLog\code\betaInfo.pkl", 'rb')
df1 = pickle.load(F1)
df1["date"] = df1["date"].apply(lambda x: int(x.strftime("%Y%m%d")))
df1 = df1[df1["date"] == date]
re1 = pd.merge(df1.loc[:, ["ID", "beta_60"]], re1, on=["ID"], how="inner")
re1["buyAlpha"] = np.where(re1["orderDirection"] == 1, re1["buyRet"] - re1["indexRet"]*re1["beta_60"], np.nan)
re1["sellAlpha"] = np.where(re1["orderDirection"] == -1, re1["sellRet"] + re1["indexRet"]*re1["beta_60"], np.nan)




print(orderLog)