import os
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

statsData = {}
for col in ['broker', 'date', 'secid', 'vai', 'side', 'our ntl', 'market orders', 'market ntl', 'front orders',
            'front ntl']:
    statsData[col] = []

startDate = '20200205'
endDate = '20200205'

readPath = r'\\192.168.10.30\Kevin_zhenyu\orderLog\result\marketPos'
dataPathLs = np.array(glob.glob(os.path.join(readPath, 'marketPos_***.pkl')))
dateLs = np.array([os.path.basename(i).split('.')[0].split('_')[1] for i in dataPathLs])
dataPathLs = dataPathLs[(dateLs >= startDate) & (dateLs <= endDate)]
checkData = []
for path in dataPathLs:
    thisDate = os.path.basename(path).split('.')[0].split('_')[1]
    data = pd.read_pickle(path)
    data['date'] = thisDate
    checkData += [data]
checkData = pd.concat(checkData).reset_index(drop=True)

checkData['brokerLs'] = np.where(checkData['brokerLs'].isnull(), ' ', checkData['brokerLs'])
checkData['brokerLs'] = checkData['brokerLs'].apply(lambda x: x.split(','))
checkData['firstBroker'] = checkData['brokerLs'].apply(lambda x: x[0])
# checkData['firstBroker'] = np.where(checkData['brokerLs'].isnull(), [' '], checkData['brokerLs'])
# checkData['firstBroker'] = checkData['firstBroker'].apply(lambda x: x[0])
for date, dateData in checkData.groupby(['date']):
    for group, groupData in dateData.groupby(['group']):
        groupData = groupData.reset_index(drop=True)
        ### ignore cases where brokerNum >= 2 (around 2.5% data)
        if groupData[groupData['brokerNum'] >= 2].shape[0] == 0:
            groupData['index'] = groupData.index.values
            groupData['lastIx'] = groupData.groupby(['firstBroker'])['index'].transform('last')
            groupData['isOurs'] = np.where((groupData['index'] == groupData['lastIx']) & (groupData['brokerNum'] == 1),
                                           1, 0)
            brokerLs = groupData[groupData['brokerNum'] == 1]['firstBroker'].unique()
            for broker in brokerLs:
                totalOrders = groupData[groupData['isOurs'] == 0].shape[0]
                totalSize = groupData[groupData['isOurs'] == 0]['OrderNtl'].sum()
                ix = groupData[(groupData['firstBroker'] == broker) & (groupData['isOurs'] == 1)].index.values[0]
                frontData = groupData.iloc[:ix]
                frontData = frontData[frontData['isOurs'] == 0]
                frontSize = frontData['OrderNtl'].sum()
                statsData['broker'].append(broker)
                statsData['date'].append(date)
                statsData['secid'].append(groupData['secid'].values[0])
                statsData['vai'].append(groupData['vai'].values[0])
                statsData['side'].append(groupData['Side'].values[0])
                statsData['our ntl'].append(
                    groupData[(groupData['firstBroker'] == broker) & (groupData['isOurs'] == 1)]['OrderNtl'].values[0])
                statsData['market orders'].append(totalOrders)
                statsData['market ntl'].append(totalSize)
                statsData['front orders'].append(frontData.shape[0])
                statsData['front ntl'].append(frontSize)

statsData = pd.DataFrame(statsData)
statsData['exchange'] = np.where(statsData['secid'] >= 2000000, 'SZE', 'SSE')
statsData['count'] = statsData.groupby(['broker', 'exchange', 'market orders'])['secid'].transform('count')