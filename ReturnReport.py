import os
import glob
import datetime
import numpy as np
import pandas as pd
import pickle

import os
import glob
import datetime
import numpy as np
import pandas as pd
import pickle


startDate = '20200430'
endDate = '20200430'

readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\equityTradeLogs'
dataPathLs = np.array(glob.glob(os.path.join(readPath, 'speedCompare***.csv')))
dateLs = np.array([os.path.basename(i).split('_')[1].split('.')[0] for i in dataPathLs])
dataPathLs = dataPathLs[(dateLs >= startDate) & (dateLs <= endDate)]
dateLs = dateLs[(dateLs >= startDate) & (dateLs <= endDate)]



def findTmValue(clockLs, tm, method='L', buffer=0):
    maxIx = len(clockLs)
    orignIx = np.arange(maxIx)
    if method == 'F':
        ix = np.searchsorted(clockLs, clockLs + (tm - buffer))
        ## if target future index is next tick, mask
        mask = (orignIx == (ix - 1)) | (orignIx == ix) | (ix == maxIx)
    elif method == 'L':
        ## if target future index is last tick, mask
        ix = np.searchsorted(clockLs, clockLs - (tm - buffer))
        ix = ix - 1
        ix[ix < 0] = 0
        mask = (orignIx == ix) | ((clockLs - (tm - buffer)).values < clockLs.values[0])
    ix[mask] = -1
    return ix


mdData1 = pd.DataFrame()
mdindex1 = pd.DataFrame()
order1 = pd.DataFrame()

# 1. load mdLog data

for date in dateLs:

    readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\equityTradeLogs'
    orderLog = pd.read_csv(os.path.join(readPath, 'speedCompare_%s.csv' % date))
    orderLog['order'] = orderLog.groupby(['date', 'accCode', 'secid', 'vai']).grouper.group_info[0]
    orderLog['orderNtl'] = orderLog['absOrderSize'] * orderLog['orderPrice']
    orderLog['exchange'] = np.where(orderLog['secid'] >= 2000000, 'SZE', 'SSE')
    orderLog['tradeNtl'] = np.where(orderLog['updateType'] == 4,
                                    orderLog['tradePrice'] * orderLog['absFilledThisUpdate'], 0)
    orderLog['startClock'] = orderLog.groupby(['order'])['clockAtArrival'].transform('first')
    orderLog['duration'] = orderLog['clockAtArrival'] - orderLog['startClock']

    ### make sure no order has shares > 80w or notional > 800w
    if orderLog[orderLog['absOrderSize'] > 800000].shape[0] > 0:
        print('some order quantity are > 80w')
        print(orderLog[orderLog['absOrderSize'] > 800000][
                    ['date', 'accCode', 'secid', 'vai', 'absOrderSize', 'orderPrice',
                     'orderNtl', 'orderDirection', 'clock']])
    if orderLog[orderLog['orderNtl'] > 8000000].shape[0] > 0:
        print('some order ntl are > 800w')
        print(orderLog[orderLog['absOrderSize'] > 8000000][
                    ['date', 'accCode', 'secid', 'vai', 'absOrderSize', 'orderPrice',
                     'orderNtl', 'orderDirection']])
    ### make sure same direction in same colo_broker
    orderLog['directNum'] = orderLog.groupby(['date', 'secid', 'vai'])['orderDirection'].transform('nunique')
    if len(orderLog[orderLog['directNum'] != 1]) > 0:
        print('opposite direction for same date, same secid, same vai')
        print(orderLog[orderLog['directNum'] != 1][['date', 'accCode', 'secid', 'vai', 'orderDirection']])
        orderLog = orderLog[orderLog['directNum'] == 1]
    assert ((orderLog.groupby(['date', 'secid', 'vai'])['orderDirection'].nunique() == 1).all() == True)
    ## make sure each account, secid, vai only has one insertion
    a = orderLog[orderLog['updateType'] == 0].groupby(['date', 'accCode', 'secid', 'vai', 'order'])[
        'clockAtArrival'].count()
    if len(a[a > 1]) > 0:
        print('more than one insertion at same time')
        a = a[a > 1].reset_index()
        print(a)
        orderLog = orderLog[~(orderLog['order'].isin(a['order'].unique()))]


    ### make sure there is no unexpected updateType
    def getTuple(x):
        return tuple(i for i in x)


    checkLog = orderLog[~((orderLog['updateType'] == 4) & (orderLog.groupby(['order'])['updateType'].shift(-1) == 4))]
    checkLog = checkLog.groupby(['order'])['updateType'].apply(lambda x: getTuple(x)).reset_index()
    checkLog['status'] = np.where(
        checkLog['updateType'].isin([(0, 2, 4), (0, 2, 1, 4), (0, 2, 1, 2, 4), (0, 2, 4, 1, 4), (0, 4), (0, 4, 1, 4)]),
        0,
        np.where(checkLog['updateType'].isin([(0, 2, 4, 1, 3), (0, 2, 4, 1, 4, 3), (0, 2, 1, 4, 3), (0, 4, 1, 3)]), 1,
                 np.where(checkLog['updateType'] == (0, 2, 1, 3), 2,
                          np.where(checkLog['updateType'].isin([(0, 3)]), 3,
                                   np.where(checkLog['updateType'].isin([(0,), (0, 2), (0, 2, 1)]), 4, 5)))))
    print(checkLog[checkLog['status'] == 5])
    orderLog = pd.merge(orderLog, checkLog[['order', 'status']], how='left', on=['order'], validate='many_to_one')
    orderLog = orderLog[orderLog['status'].isin([0, 1, 2])].reset_index(drop=True)
    ### check status==0 got all traded
    a = orderLog[orderLog['status'] == 0]
    a = a.groupby(['order'])[['absOrderSizeCumFilled', 'absOrderSize']].max().reset_index()
    a.columns = ['order', 'filled', 'total']
    print('in total trade, any fill != total cases')
    print(a[a['filled'] != a['total']])
    if a[a['filled'] != a['total']].shape[0] > 0:
        removeOrderLs = a[a['filled'] != a['total']]['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]
    ### check status==1 got partial traded
    a = orderLog[orderLog['status'] == 1]
    a = a.groupby(['order'])[['absOrderSizeCumFilled', 'absOrderSize']].max().reset_index()
    a.columns = ['order', 'filled', 'total']
    print('in partial trade, any fill >= total or fill is 0 cases for updateType 4')
    print(a[(a['filled'] >= a['total']) | (a['filled'] == 0)])
    if a[(a['filled'] >= a['total']) | (a['filled'] == 0)].shape[0] > 0:
        removeOrderLs = a[(a['filled'] >= a['total']) | (a['filled'] == 0)]['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]
    ### check if any cancellation within 1 sec
    a = orderLog[(orderLog['updateType'] == 1) & (orderLog['duration'] < 1e6)]
    print('any cancellation within 1 sec')
    print(a)
    if a.shape[0] > 0:
        removeOrderLs = a['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]

    orderLog = orderLog.sort_values(by=['date', 'secid', 'vai', 'accCode', 'clockAtArrival']).reset_index(drop=True)
    orderLog = orderLog[orderLog["updateType"].isin([0, 4])]
    orderLog = orderLog.rename(columns={'vai': "cum_volume"})
    li_st = pd.read_csv(r'F:\orderLog\code\indexCatInfo.csv', encoding="utf-8")
    orderLog = pd.merge(orderLog, li_st, left_on="secid", right_on="secid")
    orderLog['index1'] = np.where(orderLog["index"] == 'CSIRest', 'CSI1000', orderLog["index"])
    stock_list = orderLog['secid'].unique()
    order1 = pd.concat([orderLog, order1])


    readPath = r'\\192.168.10.30\Kevin_zhenyu\rawData\logs_%s_zs_92_01_day_data' % date
    mdDataSHPath = glob.glob(os.path.join(readPath, 'mdLog_SH***.csv'))[-1]
    mdDataSH = pd.read_csv(mdDataSHPath)
    mdDataSH['secid'] = mdDataSH['StockID'] + 1000000

    mdindex = mdDataSH[mdDataSH['StockID'] < 600000]
    mdindex['time'] = mdindex.time.str.slice(0, 2) + mdindex.time.str.slice(3, 5) + mdindex.time.str.slice(6, 8) + '000'
    mdindex['time'] = mdindex['time'].astype('int64')
    mdindex['time'] = mdindex.groupby(['StockID'])['time'].cummax()
    mdindex['max_cum_volume'] = mdindex.groupby(['StockID'])['cum_volume'].cummax()
    mdindex = mdindex[mdindex["StockID"] != 16]
    mdindex["index1"] = np.where(mdindex["StockID"] == 300, "IF", np.where(mdindex["StockID"] == 905, "IC", "CSI1000"))
    mdindex = mdindex[(mdindex['cum_volume'] > 0) & (mdindex['cum_volume'] == mdindex['max_cum_volume'])]
    mdindex = mdindex[['StockID', 'clockAtArrival', 'sequenceNo', 'time', 'close', "index1"]]

    mdDataSH = mdDataSH[mdDataSH["secid"].isin(stock_list)]
    mdDataSH = mdDataSH[mdDataSH['StockID'] >= 600000]
    mdDataSH['time'] = mdDataSH.time.str.slice(0, 2) + mdDataSH.time.str.slice(3, 5) + mdDataSH.time.str.slice(6,
                                                                                                               8) + '000'
    mdDataSH['time'] = mdDataSH['time'].astype('int64')
    mdDataSH['time'] = mdDataSH.groupby(['secid'])['time'].cummax()
    mdDataSH['max_cum_volume'] = mdDataSH.groupby(['StockID'])['cum_volume'].cummax()
    mdDataSH = mdDataSH[(mdDataSH['cum_volume'] > 0) & (mdDataSH['cum_volume'] == mdDataSH['max_cum_volume'])]
    mdDataSH = mdDataSH[
        ['secid', 'clockAtArrival', 'sequenceNo', 'time', 'cum_volume', 'bid1p', 'ask1p', 'bid1q', 'ask1q', 'bid5q',
         'ask5q']]

    mdDataSZPath = glob.glob(os.path.join(readPath, 'mdLog_SZ***.csv'))[-1]
    mdDataSZ = pd.read_csv(mdDataSZPath)
    mdDataSZ['secid'] = mdDataSZ['StockID'] + 2000000
    mdDataSZ = mdDataSZ[mdDataSZ["secid"].isin(stock_list)]

    mdDataSZ['time'] = mdDataSZ.time.str.slice(0, 2) + mdDataSZ.time.str.slice(3, 5) + mdDataSZ.time.str.slice(6,
                                                                                                               8) + '000'
    mdDataSZ['time'] = mdDataSZ['time'].astype('int64')
    mdDataSZ['time'] = mdDataSZ.groupby(['secid'])['time'].cummax()
    mdDataSZ['max_cum_volume'] = mdDataSZ.groupby(['StockID'])['cum_volume'].cummax()
    mdDataSZ = mdDataSZ[(mdDataSZ['cum_volume'] > 0) & (mdDataSZ['cum_volume'] == mdDataSZ['max_cum_volume'])]
    mdDataSZ = mdDataSZ[
        ['secid', 'clockAtArrival', 'sequenceNo', 'time', 'cum_volume', 'bid1p', 'ask1p', 'bid1q', 'ask1q', 'bid5q',
         'ask5q']]

    mdData = pd.concat([mdDataSH, mdDataSZ]).reset_index(drop=True)
    mdData = pd.merge(mdData, li_st, left_on="secid", right_on="secid")
    li_st = li_st[li_st["secid"].isin(stock_list)]
    li_st['index1'] = np.where(li_st["index"] == 'CSIRest', 'CSI1000', li_st["index"])
    list1 = li_st[li_st["index1"] == "IC"]["secid"].values
    list2 = li_st[li_st["index1"] == "IF"]["secid"].values
    list3 = li_st[li_st["index1"] == "CSI1000"]["secid"].values

    d1 = mdData[mdData["secid"].isin(list1)]
    d2 = mdindex[mdindex["index1"] == "IC"]
    d1 = pd.concat([d1, d2.loc[:, ["sequenceNo", "clockAtArrival", "close"]]]).sort_values(by=["sequenceNo", "clockAtArrival"])
    d1["close"] = d1["close"].ffill().bfill()
    data = d1[~d1["secid"].isnull()]
    d1 = mdData[mdData["secid"].isin(list2)]
    d2 = mdindex[mdindex["index1"] == "IF"]
    d1 = pd.concat([d1, d2.loc[:, ["sequenceNo", "clockAtArrival", "close"]]]).sort_values(by=["sequenceNo", "clockAtArrival"])
    d1["close"] = d1["close"].ffill().bfill()
    data = pd.concat([data, d1[~d1["secid"].isnull()]])
    d1 = mdData[mdData["secid"].isin(list3)]
    d2 = mdindex[mdindex["index1"] == "CSI1000"]
    d1 = pd.concat([d1, d2.loc[:, ["sequenceNo", "clockAtArrival", "close"]]]).sort_values(by=["sequenceNo", "clockAtArrival"])
    d1["close"] = d1["close"].ffill().bfill()
    data = pd.concat([data, d1[~d1["secid"].isnull()]])
    mdData = data

    mdData = mdData.sort_values(by=['secid', 'sequenceNo']).reset_index(drop=True)
    mdData['safeBid1p'] = np.where(mdData['bid1p'] == 0, mdData['ask1p'], mdData['bid1p'])
    mdData['safeAsk1p'] = np.where(mdData['ask1p'] == 0, mdData['bid1p'], mdData['ask1p'])
    mdData['adjMid'] = (mdData['safeBid1p'] * mdData['ask1q'] + mdData['safeAsk1p'] * mdData['bid1q']) / (
            mdData['bid1q'] + mdData['ask1q'])
    mdData['session'] = np.where(mdData['time'] >= 130000000, 1, 0)

    mdData = mdData.reset_index(drop=True)
    groupAllData = mdData.groupby(['secid', 'session'])
    mdData['sessionStartCLA'] = groupAllData['clockAtArrival'].transform('min')
    mdData['relativeClock'] = mdData['clockAtArrival'] - mdData['sessionStartCLA']
    mdData['trainFlag'] = np.where(mdData['relativeClock'] > 179.5 * 1e6, 1, 0)
    mdData['index1'] = mdData.index.values
    mdData['sessionStartIx'] = groupAllData['index1'].transform('min')
    for tm in [90]:
        tmCol = 'F{}s_ix'.format(tm)
        mdData[tmCol] = groupAllData['relativeClock'].transform(
            lambda x: findTmValue(x, tm * 1e6, 'F', 5 * 1e5)).astype(int)
    for tm in [90]:
        tmIx = mdData['F{}s_ix'.format(tm)].values + mdData['sessionStartIx'].values
        adjMid_tm = mdData['adjMid'].values[tmIx]
        adjMid_tm[mdData['F{}s_ix'.format(tm)].values == -1] = np.nan
        mdData['adjMid_F{}s'.format(tm)] = adjMid_tm
        adjMid_tm1 = mdData['close'].values[tmIx]
        adjMid_tm1[mdData['F{}s_ix'.format(tm)].values == -1] = np.nan
        mdData['close_F{}s'.format(tm)] = adjMid_tm1
    mdData["date"] = date

    del mdDataSH
    del mdDataSZ
    mdData = mdData.loc[:, ['date', "secid", "clockAtArrival", "cum_volume", "adjMid", "adjMid_F90s",
                            "close", "close_F90s", "index"]]

    mdData1 = pd.concat([mdData, mdData1])

# 2. find each order and trade's forward 90s stock adjMid price
mdData1["date"] = mdData1["date"].astype(int)
mdData1["isOrder"] = 0
order1["isOrder"] = 1

re = pd.concat([order1, mdData1.loc[:, ['date', "secid", "cum_volume", "clockAtArrival", "adjMid",
                                        "adjMid_F90s", 'isOrder', "close", "close_F90s", "index"]]])
re = re.sort_values(by=["date", "secid", "cum_volume", "isOrder"]).reset_index(drop=True)
re["adjMid_F90s"] = re.groupby(["date", "secid"])["adjMid_F90s"].bfill()
re["close_F90s"] = re.groupby(["date", "secid"])["close_F90s"].bfill()
re["close"] = re.groupby(["date", "secid"])["close"].ffill()
re = re[re["isOrder"] == 1]
re["buyRet"] = np.where((re["orderDirection"] == 1) & (re["updateType"] == 4), re["adjMid_F90s"] / re["tradePrice"] - 1, np.nan)
re["sellRet"] = np.where((re["orderDirection"] == -1) & (re["updateType"] == 4), re["tradePrice"] / re["adjMid_F90s"] - 1, np.nan)
F1 = open(r"\\192.168.10.30\Kevin_zhenyu\orderLog\betaInfo\betaInfo_20191101_20200430.pkl", 'rb')
df1 = pickle.load(F1)
df1["date"] = df1["date"].apply(lambda x: int(x.strftime("%Y%m%d")))
df1["indexCat"] = np.where(df1["indexCat"] == 1000300, "IF", np.where(
    df1["indexCat"] == 1000852, "CSI1000", "IC"
))
re = pd.merge(df1, re, left_on=["date", "ID", "indexCat"], right_on=["date", "secid", "index1"], how="inner")
savePath = r'F:\orderLog\result\90s return'
re[re["date"] == int(startDate)].to_csv(os.path.join(savePath, 'OrderLog1_%s.csv' % int(startDate)), index=False)

# 3.  find corresponding index return

# re1[re1["date"] == int(endDate)].to_csv(os.path.join(savePath, 'OrderLog1_%s.csv' % int(endDate)), index=False)
# re1 = re1[re1["updateType"] == 4]
# re1["buyAlpha"] = np.where(re1["orderDirection"] == 1, re1["buyRet"] - re1["indexRet"] * re1["beta_60"], np.nan)
# re1["sellAlpha"] = np.where(re1["orderDirection"] == -1, re1["sellRet"] + re1["indexRet"] * re1["beta_60"], np.nan)
# print(re1)
#
# # 4. generate daily weighted return and alpha
#
# re1["buyNtl"] = np.where(~re1["buyRet"].isnull(), re1["tradeNtl"], np.nan)
# re1["sellNtl"] = np.where(~re1["sellRet"].isnull(), re1["tradeNtl"], np.nan)
# re1["sumbuyNtl"] = re1.groupby(["date", "exchange", "index", "colo"])["buyNtl"].transform(sum)
# re1["sumsellNtl"] = re1.groupby(["date", "exchange", "index", "colo"])["sellNtl"].transform(sum)
#
# re1["sumsellRet"] = re1["tradeNtl"] * re1["sellRet"]
# re1["sumsellRet"] = re1.groupby(["date", "exchange", "index", "colo"])["sumsellRet"].transform(sum)
#
# re1["sumbuyAlpha"] = re1["tradeNtl"] * re1["buyAlpha"]
# re1["sumbuyAlpha"] = re1.groupby(["date", "exchange", "index", "colo"])["sumbuyAlpha"].transform(sum)
#
# re1["sumsellAlpha"] = re1["tradeNtl"] * re1["sellAlpha"]
# re1["sumsellAlpha"] = re1.groupby(["date", "exchange", "index", "colo"])["sumsellAlpha"].transform(sum)
#
# re1["sumbuyRet"] = re1["tradeNtl"] * re1["buyRet"]
# re1["sumbuyRet"] = re1.groupby(["date", "exchange", "index", "colo"])["sumbuyRet"].transform(sum)
#
# re1["tnw_buyRet"] = re1["sumbuyRet"] / re1["sumbuyNtl"]
# re1["tnw_sellRet"] = re1["sumsellRet"] / re1["sumsellNtl"]
# re1["tnw_buyAlpha"] = re1["sumbuyAlpha"] / re1["sumbuyNtl"]
# re1["tnw_sellAlpha"] = re1["sumsellAlpha"] / re1["sumsellNtl"]
#
# save1 = re1[re1["date"] == int(startDate)].groupby(["date", "exchange", "index", "colo"])[
#     "tnw_buyRet", "tnw_buyAlpha", "tnw_sellRet", "tnw_sellAlpha"].first().reset_index()
# save2 = re1[re1["date"] == int(endDate)].groupby(["date", "exchange", "index", "colo"])[
#     "tnw_buyRet", "tnw_buyAlpha", "tnw_sellRet", "tnw_sellAlpha"].first().reset_index()
# savePath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\result\90s return'
# save1.to_csv(os.path.join(savePath, '90sReturn_%s.csv' % int(startDate)), index=False)
# save2.to_csv(os.path.join(savePath, '90sReturn_%s.csv' % int(endDate)), index=False)
#
# for col in ["tnw_buyRet", "tnw_sellRet", "tnw_buyAlpha", "tnw_sellAlpha"]:
#     re1[col] = re1[col].apply(lambda x: '%.4f' % (x))
# print(re1.groupby(["date", "exchange", "index", "colo"])[
#          "tnw_buyRet", "tnw_buyAlpha", "tnw_sellRet", "tnw_sellAlpha"].first())