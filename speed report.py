import os
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startTm = datetime.datetime.now()

startDate = '20200427'
endDate = '20200508'

readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\equityTradeLogs'
dataPathLs = np.array(glob.glob(os.path.join(readPath, 'speedCompare***.csv')))
dateLs = np.array([os.path.basename(i).split('_')[1].split('.')[0] for i in dataPathLs])
dataPathLs = dataPathLs[(dateLs >= startDate) & (dateLs <= endDate)]
dateLs = dateLs[(dateLs >= startDate) & (dateLs <= endDate)]

for thisDate in dateLs[6:]:

    readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\equityTradeLogs'
    orderLog = pd.read_csv(os.path.join(readPath, 'speedCompare_%s.csv' % thisDate))
    for col in ['clockAtArrival', 'secid', 'updateType', 'vai', 'absFilledThisUpdate', 'orderDirection', 'absOrderSize',
                'absOrderSizeCumFilled', 'date', 'accCode', 'mse']:
        orderLog[col] = orderLog[col].astype('int64')

    orderLog = orderLog.sort_values(by=['date', 'secid', 'vai', 'accCode', 'clockAtArrival']).reset_index(drop=True)
    orderLog = orderLog[orderLog["secid"] >= 1000000]

    targetStock = orderLog['secid'].unique()
    targetStock = np.array([int(str(i)[1:]) for i in targetStock])
    targetStockSZ = sorted(targetStock[targetStock < 600000])
    targetStockSH = sorted(targetStock[targetStock >= 600000])

    orderLog['clock'] = orderLog['clockAtArrival'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1e6))
    orderLog['broker'] = orderLog['accCode'] // 100
    orderLog['colo_account'] = orderLog['colo'].str[:2] + '_' + orderLog['accCode'].astype('str')
    orderLog['order'] = orderLog.groupby(['date', 'accCode', 'secid', 'vai']).grouper.group_info[0]
    orderLog['group'] = orderLog.groupby(['date', 'secid', 'vai']).grouper.group_info[0]
    orderLog['startClock'] = orderLog.groupby(['order'])['clockAtArrival'].transform('first')
    orderLog['duration'] = orderLog['clockAtArrival'] - orderLog['startClock']
    orderLog['orderPrice'] = orderLog['orderPrice'].apply(lambda x: round(x, 2))
    orderLog['tradePrice'] = orderLog['tradePrice'].apply(lambda x: round(x, 2))
    orderLog = orderLog.copy()

    ### make sure no order has shares > 80w or notional > 800w
    orderLog['orderNtl'] = orderLog['absOrderSize'] * orderLog['orderPrice']

    ### make sure same direction in same colo_broker
    orderLog['directNum'] = orderLog.groupby(['date', 'secid', 'vai'])['orderDirection'].transform('nunique')
    if len(orderLog[orderLog['directNum'] != 1]) > 0:
        orderLog = orderLog[orderLog['directNum'] == 1]

    assert ((orderLog.groupby(['date', 'secid', 'vai'])['orderDirection'].nunique() == 1).all() == True)

    ## make sure each account, secid, vai only has one insertion
    a = orderLog[orderLog['updateType'] == 0].groupby(['date', 'accCode', 'secid', 'vai', 'order'])[
        'clockAtArrival'].count()
    if len(a[a > 1]) > 0:
        a = a[a > 1].reset_index()
        orderLog = orderLog[~(orderLog['order'].isin(a['order'].unique()))]

    orderLog['isMsg'] = np.where(orderLog['updateType'] == 0,
                                 np.where(orderLog['mse'] == 100, 1, 0), np.nan)
    orderLog['isMsg'] = orderLog.groupby(['order'])['isMsg'].ffill()
    placeSZE = orderLog[(orderLog['secid'] >= 2000000) & (orderLog['updateType'] == 0)]


    ### make sure there is no unexpected updateType
    def getTuple(x):
        return tuple(i for i in x)


    checkLog = orderLog[~((orderLog['updateType'] == 4) & (orderLog.groupby(['order'])['updateType'].shift(-1) == 4))]
    checkLog = checkLog.groupby(['order'])['updateType'].apply(lambda x: getTuple(x)).reset_index()
    checkLog['status'] = np.where(
        checkLog['updateType'].isin([(0, 2, 4), (0, 2, 2, 4), (0, 2, 2, 1, 4), (0, 2, 1, 4), (0, 2, 1, 2, 4), (0, 2, 4, 1, 4), (0, 4), (0, 4, 1, 4), (0, 2, 2, 4, 1, 4), (0, 4, 2, 4)]),
        0,
        np.where(checkLog['updateType'].isin([(0, 2, 4, 1, 3), (0, 2, 4, 1, 4, 3), (0, 2, 1, 4, 3), (0, 4, 1, 3), (0, 2, 2, 4, 1, 4, 3),
                                              (0, 2, 2, 4, 1, 3), (0, 2, 2, 1, 4, 3), (0, 4, 2, 4, 1, 3), (0, 4, 2, 1, 3), (0, 2, 4, 2, 1, 3)]), 1,
                 np.where(checkLog['updateType'].isin([(0, 2, 1, 3), (0, 2, 2, 1, 3)]), 2,
                          np.where(checkLog['updateType'].isin([(0, 3)]), 3,
                                   np.where(checkLog['updateType'].isin([(0,), (0, 2), (0, 2, 1)]), 4, 5)))))
    print(set(checkLog["updateType"].unique()) - set([(0, 2, 4), (0, 2, 2, 4), (0, 2, 2, 1, 4), (0, 2, 1, 4), (0, 2, 1, 2, 4),
                                                      (0, 2, 4, 1, 4), (0, 4), (0, 4, 1, 4), (0, 2, 2, 4, 1, 4), (0, 4, 2, 4),
                                                      (0, 2, 4, 1, 3), (0, 2, 4, 1, 4, 3), (0, 2, 1, 4, 3), (0, 4, 1, 3),
                                                      (0, 2, 2, 4, 1, 4, 3), (0, 2, 2, 4, 1, 3), (0, 2, 2, 1, 4, 3),
                                                      (0, 4, 2, 4, 1, 3), (0, 4, 2, 1, 3), (0, 2, 1, 3), (0, 2, 2, 1, 3),
                                                      (0, 3), (0,), (0, 2), (0, 2, 1), (0, 2, 4, 2, 1, 3)]))

    orderLog = pd.merge(orderLog, checkLog[['order', 'status']], how='left', on=['order'], validate='many_to_one')
    orderLog = orderLog[orderLog['status'].isin([0, 1, 2])].reset_index(drop=True)

    ### check status==0 got all traded
    a = orderLog[orderLog['status'] == 0]
    a = a.groupby(['order'])[['absOrderSizeCumFilled', 'absOrderSize']].max().reset_index()
    a.columns = ['order', 'filled', 'total']
    if a[a['filled'] != a['total']].shape[0] > 0:
        removeOrderLs = a[a['filled'] != a['total']]['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]

    ### check status==1 got partial traded
    a = orderLog[orderLog['status'] == 1]
    a = a.groupby(['order'])[['absOrderSizeCumFilled', 'absOrderSize']].max().reset_index()
    a.columns = ['order', 'filled', 'total']
    if a[(a['filled'] >= a['total']) | (a['filled'] == 0)].shape[0] > 0:
        removeOrderLs = a[(a['filled'] >= a['total']) | (a['filled'] == 0)]['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]

    ### check if any cancellation within 1 sec
    a = orderLog[(orderLog['updateType'] == 1) & (orderLog['duration'] < 1e6)]
    if a.shape[0] > 0:
        removeOrderLs = a['order'].unique()
        orderLog = orderLog[~(orderLog['order'].isin(removeOrderLs))]

    orderLog = orderLog.sort_values(by=['date', 'secid', 'vai', 'accCode', 'clockAtArrival']).reset_index(drop=True)

    orderLog['exchange'] = np.where(orderLog['secid'] >= 2000000, 'SZE', 'SSE')
    orderLog['orderNtl'] = orderLog['orderPrice'] * orderLog['absOrderSize']
    orderLog['tradeNtl'] = np.where(orderLog['updateType'] == 4,
                                    orderLog['tradePrice'] * orderLog['absFilledThisUpdate'], 0)
    orderLog = orderLog[orderLog['secid'] >= 2000000].reset_index(drop=True)

    readPath = r'F:\orderLog\mdData'
    rawMsgDataSZ = pd.read_pickle(os.path.join(readPath, 'mdLog_msg_%s.pkl' % thisDate))
    orderDataSZ = rawMsgDataSZ[rawMsgDataSZ['ExecType'] == '2'][
        ['SecurityID', 'ApplSeqNum', 'clockAtArrival', 'sequenceNo', 'Side', 'OrderQty', 'Price',
         'cum_volume']].reset_index(drop=True)
    orderDataSZ['updateType'] = 0
    tradeDataSZ = pd.concat([rawMsgDataSZ[rawMsgDataSZ['ExecType'] == 'F'][
                                 ['SecurityID', 'BidApplSeqNum', 'clockAtArrival', 'sequenceNo', 'TradePrice',
                                  'TradeQty', 'cum_volume']],
                             rawMsgDataSZ[rawMsgDataSZ['ExecType'] == 'F'][
                                 ['SecurityID', 'OfferApplSeqNum', 'clockAtArrival', 'sequenceNo', 'TradePrice',
                                  'TradeQty', 'cum_volume']]], sort=False)
    tradeDataSZ['ApplSeqNum'] = np.where(tradeDataSZ['BidApplSeqNum'].isnull(), tradeDataSZ['OfferApplSeqNum'],
                                         tradeDataSZ['BidApplSeqNum'])
    tradeDataSZ['Side'] = np.where(tradeDataSZ['BidApplSeqNum'].isnull(), 2, 1)
    tradeDataSZ = tradeDataSZ[
        ['SecurityID', 'ApplSeqNum', 'clockAtArrival', 'sequenceNo', 'Side', 'TradePrice', 'TradeQty', 'cum_volume']]
    tradeDataSZ['updateType'] = 4
    cancelDataSZ = rawMsgDataSZ[rawMsgDataSZ['ExecType'] == '4'][
        ['SecurityID', 'BidApplSeqNum', 'OfferApplSeqNum', 'clockAtArrival', 'sequenceNo', 'TradePrice', 'TradeQty',
         'cum_volume']].reset_index(drop=True)
    cancelDataSZ['ApplSeqNum'] = np.where(cancelDataSZ['BidApplSeqNum'] == 0, cancelDataSZ['OfferApplSeqNum'],
                                          cancelDataSZ['BidApplSeqNum'])
    cancelDataSZ['Side'] = np.where(cancelDataSZ['BidApplSeqNum'] == 0, 2, 1)
    cancelDataSZ = cancelDataSZ[
        ['SecurityID', 'ApplSeqNum', 'clockAtArrival', 'sequenceNo', 'Side', 'TradeQty', 'cum_volume']]
    cancelDataSZ['updateType'] = 3

    msgDataSZ = pd.concat([orderDataSZ, tradeDataSZ, cancelDataSZ], sort=False)
    del orderDataSZ
    del tradeDataSZ
    del cancelDataSZ
    msgDataSZ = msgDataSZ.sort_values(by=['SecurityID', 'ApplSeqNum', 'sequenceNo']).reset_index(drop=True)

    msgDataSZ['TradePrice'] = np.where(msgDataSZ['updateType'] == 4, msgDataSZ['TradePrice'], 0)
    msgDataSZ['TradePrice'] = msgDataSZ['TradePrice'].astype('int64')
    msgDataSZ['TradeQty'] = np.where(msgDataSZ['updateType'] == 4, msgDataSZ['TradeQty'], 0)
    msgDataSZ['TradeQty'] = msgDataSZ['TradeQty'].astype('int64')
    msgDataSZ['secid'] = msgDataSZ['SecurityID'] + 2000000
    assert (msgDataSZ['ApplSeqNum'].max() < 1e8)
    msgDataSZ['StockSeqNum'] = msgDataSZ['SecurityID'] * 1e8 + msgDataSZ['ApplSeqNum']
    msgDataSZ['date'] = int(thisDate)
    msgDataSZ['startVolume'] = msgDataSZ.groupby(['StockSeqNum'])['cum_volume'].transform('first')

    ### order insertion position
    startPos = orderLog[
        (orderLog['date'] == int(thisDate)) & (orderLog['updateType'] == 0) & (orderLog['secid'] >= 2000000) & (
                    orderLog['isMsg'] == 1)]
    # here!!!!!!!!!! drop duplicates
    startPos = startPos.drop_duplicates(subset=['date', 'secid', 'vai'])

    startPos = startPos[
        ((startPos.clock.dt.time >= datetime.time(9, 33)) & (startPos.clock.dt.time <= datetime.time(11, 30))) | \
        ((startPos.clock.dt.time >= datetime.time(13, 3)) & (startPos.clock.dt.time <= datetime.time(14, 55)))]
    #     startPos = startPos[startPos.clock.dt.time < datetime.time(9, 33)]

    startPos['SecurityID'] = startPos['secid'] - 2000000
    startPos['orderDirection'] = np.where(startPos['orderDirection'] == 1, 1, 2)
    startPos['cum_volume'] = startPos['vai']
    startPos = startPos[['SecurityID', 'cum_volume', 'orderDirection', 'accCode', 'absOrderSize', 'vai', 'group']]
    startPos['stockGroup'] = startPos.groupby(['accCode', 'SecurityID']).grouper.group_info[0]
    startPos = startPos.sort_values(by=['SecurityID', 'cum_volume']).reset_index(drop=True)

    ### generate order status change data
    infoData = orderLog[(orderLog['date'] == int(thisDate)) & (orderLog['group'].isin(startPos['group'].unique())) & (
        orderLog['updateType'].isin([0, 3, 4]))].reset_index(drop=True)
    infoData['Price'] = infoData['orderPrice'].apply(lambda x: round(x * 100, 0))
    infoData['Price'] = infoData['Price'].astype('int64') * 100
    infoData['OrderQty'] = infoData['absOrderSize']
    infoData['Side'] = np.where(infoData['orderDirection'] == 1, 1, 2)
    infoData['TradePrice'] = np.where(infoData['updateType'] == 4, round(infoData['tradePrice'] * 100, 0), 0)
    infoData['TradePrice'] = infoData['TradePrice'].astype('int64') * 100
    statusInfo = infoData.groupby(['order'])['updateType'].apply(lambda x: tuple(x)).reset_index()
    statusInfo.columns = ['order', 'statusLs']
    tradePriceInfo = infoData.groupby(['order'])['TradePrice'].apply(lambda x: tuple(x)).reset_index()
    tradePriceInfo.columns = ['order', 'TradePriceLs']
    tradeQtyInfo = infoData.groupby(['order'])['absFilledThisUpdate'].apply(lambda x: tuple(x)).reset_index()
    tradeQtyInfo.columns = ['order', 'TradeQtyLs']
    infoData = infoData[infoData['updateType'] == 0]
    infoData = pd.merge(infoData, statusInfo, how='left', on=['order'], validate='one_to_one')
    infoData = pd.merge(infoData, tradePriceInfo, how='left', on=['order'], validate='one_to_one')
    infoData = pd.merge(infoData, tradeQtyInfo, how='left', on=['order'], validate='one_to_one')
    infoData['brokerNum'] = \
    infoData.groupby(['date', 'secid', 'vai', 'Price', 'OrderQty', 'Side', 'statusLs', 'TradePriceLs', 'TradeQtyLs'])[
        'colo_account'].transform('count')
    infoData['brokerLs'] = \
    infoData.groupby(['date', 'secid', 'vai', 'Price', 'OrderQty', 'Side', 'statusLs', 'TradePriceLs', 'TradeQtyLs'])[
        'colo_account'].transform(lambda x: ','.join(sorted(x.unique())))
    infoData = infoData.drop_duplicates(
        subset=['date', 'secid', 'vai', 'Price', 'OrderQty', 'Side', 'statusLs', 'TradePriceLs',
                'TradeQtyLs']).reset_index(drop=True)
    infoData = infoData[
        ['date', 'secid', 'vai', 'ars', 'group', 'Price', 'OrderQty', 'Side', 'statusLs', 'TradePriceLs', 'TradeQtyLs',
         'brokerNum', 'brokerLs', 'order']]

    ### find all orders in market that inserted with us
    checkLog = msgDataSZ[msgDataSZ['updateType'] == 0][
        ['StockSeqNum', 'SecurityID', 'cum_volume', 'sequenceNo', 'clockAtArrival', 'Side', 'OrderQty',
         'Price']].reset_index(drop=True)
    checkLog = checkLog.sort_values(by=['SecurityID', 'sequenceNo'])
    checkLog = pd.merge(checkLog, startPos, how='outer', on=['SecurityID', 'cum_volume'], validate='many_to_one')
    del startPos
    if checkLog[(~checkLog['vai'].isnull()) & (checkLog['clockAtArrival'].isnull())].shape[0] > 0:
        print('some order vai are wrong')
        removeGroup = checkLog[(~checkLog['vai'].isnull()) & (checkLog['clockAtArrival'].isnull())][
            'stockGroup'].unique()
        checkLog = checkLog[~checkLog['stockGroup'].isin(removeGroup)].reset_index(drop=True)

    assert (checkLog[(~checkLog['vai'].isnull()) & (checkLog['clockAtArrival'].isnull())].shape[0] == 0)
    checkLog['lastClockInVol'] = checkLog.groupby(['SecurityID', 'cum_volume'])['clockAtArrival'].transform('last')
    checkLog['startClock'] = np.where(
        (~checkLog['group'].isnull()) & (checkLog['clockAtArrival'] == checkLog['lastClockInVol']),
        checkLog['clockAtArrival'], np.nan)
    checkLog['group'] = checkLog['group'].ffill()
    checkLog['startClock'] = checkLog.groupby(['SecurityID', 'group'])['startClock'].transform('max')
    checkLog['endClock'] = checkLog['startClock'] + 20 * 1e3
    checkLog = checkLog[
        (checkLog['clockAtArrival'] >= checkLog['startClock']) & (checkLog['clockAtArrival'] <= checkLog['endClock'])]
    checkLog['vai'] = checkLog.groupby(['SecurityID', 'group'])['vai'].ffill()
    checkLog['orderDirection'] = checkLog.groupby(['SecurityID', 'group'])['orderDirection'].ffill()
    checkLog = checkLog[checkLog['Side'] == checkLog['orderDirection']]

    group_list = checkLog["group"].unique()

    checkLog = pd.merge(msgDataSZ, checkLog[['StockSeqNum', 'vai', 'group']], how='left', on=['StockSeqNum'],
                        validate='many_to_one')
    del msgDataSZ
    checkLog = checkLog[~checkLog['group'].isnull()]
    # 上面能不能简化成inner merge!!!!!!!!!!!!!!!!!!!!!!!!!
    statusInfo = checkLog.groupby(['StockSeqNum'])['updateType'].apply(lambda x: getTuple(x)).reset_index()
    statusInfo.columns = ['StockSeqNum', 'statusLs']
    tradePriceInfo = checkLog.groupby(['StockSeqNum'])['TradePrice'].apply(lambda x: tuple(x)).reset_index()
    tradePriceInfo.columns = ['StockSeqNum', 'TradePriceLs']
    tradeQtyInfo = checkLog.groupby(['StockSeqNum'])['TradeQty'].apply(lambda x: tuple(x)).reset_index()
    tradeQtyInfo.columns = ['StockSeqNum', 'TradeQtyLs']
    checkLog = checkLog[checkLog['updateType'] == 0]
    checkLog = pd.merge(checkLog, statusInfo, how='left', on=['StockSeqNum'], validate='one_to_one')
    checkLog = pd.merge(checkLog, tradePriceInfo, how='left', on=['StockSeqNum'], validate='one_to_one')
    checkLog = pd.merge(checkLog, tradeQtyInfo, how='left', on=['StockSeqNum'], validate='one_to_one')

    infoData = infoData[infoData["group"].isin(group_list)]

    checkLog = pd.merge(checkLog, infoData, how='outer',
                        on=['date', 'secid', 'group', 'vai', 'Price', 'OrderQty', 'Side', 'statusLs', 'TradePriceLs',
                            'TradeQtyLs'], validate='many_to_one')
    checkLog = checkLog.sort_values(by=['date', 'secid', 'vai', 'sequenceNo']).reset_index(drop=True)
    ### orderType 1 orders have 0 order price, replace 0 with group price
    checkLog['groupPrice'] = checkLog.groupby(['group'])['Price'].transform('median')
    checkLog['Price'] = np.where(checkLog['Price'] == 0, checkLog['groupPrice'], checkLog['Price'])
    checkLog['OrderNtl'] = checkLog['Price'] * checkLog['OrderQty'] / 10000

    savePath = r'F:\orderLog\result\marketPos'
    checkLog.reset_index(drop=True).to_pickle(os.path.join(savePath, 'marketPos_%s.pkl' % thisDate))

print(datetime.datetime.now() - startTm)