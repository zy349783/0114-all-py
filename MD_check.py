import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter
import collections
from decimal import Decimal, getcontext, ROUND_HALF_UP
logSH = pd.read_csv('E:\\compare\\58\\mdLog_SH_20200107_0845.csv',
                    encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "source", "StockID",
                                              "exchange", "time", "cum_volume", "cum_amount", "close",
                                              "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q",
                                              "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                              "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
                                              "ask4q", "ask5q", "openPrice"]]
logSH1 = pd.read_csv('E:\\compare\\zt_58_0107\\Logs\\mdLog_SH_20200107_0858.csv',
                     encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "source", "StockID",
                                               "exchange", "time", "cum_volume", "cum_amount", "close",
                                               "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q",
                                               "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                               "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
                                               "ask4q", "ask5q", "openPrice"]]

round_context = getcontext()
round_context.rounding = ROUND_HALF_UP

def c_round(x, digits, precision=5):
    tmp = round(Decimal(x), precision)
    return float(tmp.__round__(digits))

# 1. data format
# 1.1 in general
print("original data source type: ")
print(logSH["source"].unique())
print("new data source type: ")
print(logSH1["source"].unique())
print(logSH["time"].unique())
print(logSH1["time"].unique())
logSH["time"] = logSH["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
#logSH1["time"] = logSH1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
logSH1["time"] = logSH1["time"].apply(lambda x: int(x.replace(':', "") + "000"))
print(logSH[(logSH["StockID"] == 600004) & (logSH["time"] > 100000000)])
print(logSH1[(logSH1["StockID"] == 600004) & (logSH1["time"] > 100000000)])
# 1.2 index data ("close")
in_dex = logSH[logSH["source"] == 5]["StockID"].unique()
print(logSH[(logSH["StockID"] == in_dex[0]) & (logSH["time"] > 93000000)].head())
print(logSH1[(logSH1["StockID"] == in_dex[0]) & (logSH1["time"] > 93000000)].head())
print("Baseline Index Data collected from 9:30 to 14:57: " + str(len(logSH[(logSH["StockID"] == in_dex[0])
                                                     & (logSH["time"] > 93000000) & (logSH["time"] < 145700000)])))
print("New Index Data collected from 9:30 to 14:57: " + str(len(logSH1[(logSH1["StockID"] == in_dex[0])
                                                & (logSH1["time"] > 93000000) & (logSH1["time"] < 145700000)])))
# 1.3 level1 & level2 data ("price", "TransactTime")
print(logSH[(logSH["StockID"] == 600355) & (logSH["time"] > 93000000) & (logSH["source"] == 3)].head())
print(logSH1[(logSH1["StockID"] == 600355) & (logSH1["time"] > 93000000) & (logSH1["source"] == 20)].head())

print(logSH[(logSH["StockID"] == 600355) & (logSH["time"] > 93000000) & (logSH["source"] == 4)].head())
print(logSH1[(logSH1["StockID"] == 600355) & (logSH1["time"] > 93000000) & (logSH1["source"] == 21)].head())

# 2. data accuracy
# 2.1 level1 data
print("start to compare level1 data")
data1 = logSH[~logSH["StockID"].isin(in_dex) & (logSH["cum_volume"] > 0) & (logSH["time"] <= 145700000)
              & (logSH["source"] == 3)]
data2 = logSH1[~logSH1["StockID"].isin(in_dex) & (logSH1["cum_volume"] > 0) & (logSH1["time"] <= 145700000)
              & (logSH1["source"] == 20)]
columns = ["cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
# drop duplicates
data1_1 = data1.groupby("StockID")[data1.columns[~data1.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data2_1 = data2.groupby("StockID")[data2.columns[~data2.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data1_1 = data1_1.drop("level_1", axis=1)
data2_1 = data2_1.drop("level_1", axis=1)
n1 = len(data1_1["StockID"].unique())
n2 = len(data2_1["StockID"].unique())
print("The number of SH level1 stocks in baseline broker is: " + str(n1))
print("The number of SH level1 stocks in new broker is: " + str(n2))
print(list(set(data1["StockID"].unique()) - set(data2["StockID"].unique())))
# merge and compare
columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
data1_1 = data1_1[data1_1["StockID"].isin(data2_1["StockID"].unique())]
test = pd.merge(data1_1, data2_1, left_on=columns, right_on=columns, how="outer")
n1 = test["sequenceNo_x"].count()
n2 = test["sequenceNo_y"].count()
len1 = len(test)
d = 0
m = 0
if n2 == len1:
    print("1. New Broker has same lv1 data as baseline broker")
    if n1 < n2:
        print("2. Baseline broker has smaller lv1 data compared with new broker:")
        print("The number of different data:")
        print(n2 - n1)
        print("The number of shared data:")
        print(n1)
        # print("3. The reason of difference:")
        # for i in range(len(test[np.isnan(test["sequenceNo_x"])]["StockID"])):
        #     if data1[(data1["StockID"] == test[np.isnan(test["sequenceNo_x"])]["StockID"].iloc[i])
        #              & (data1["time"] == test[np.isnan(test["sequenceNo_x"])]["time_y"].iloc[i])].empty == False:
        #         d = d + 1
        #     else:
        #         m = m + 1
        # print("(1) Two brokers have different data: " + str(d))
        # print("(2) Baseline broker miss some data: " + str(m))
    if n1 == n2:
        print("baseline broker has same lv1 data as new broker")

# 2.2 level2 data
print("\nstart to compare level2 data")
data1 = logSH[~logSH["StockID"].isin(in_dex) & (logSH["cum_volume"] > 0) & (logSH["time"] <= 145700000)
              & (logSH["source"] == 4)]
data1["cum_amount"] = data1["cum_amount"].apply(lambda x: c_round(x, 0))
data2 = logSH1[~logSH1["StockID"].isin(in_dex) & (logSH1["cum_volume"] > 0) & (logSH1["time"] <= 145700000)
              & (logSH1["source"] == 21)]
columns = ["cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
# drop duplicates
data1_1 = data1.groupby("StockID")[data1.columns[~data1.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data2_1 = data2.groupby("StockID")[data2.columns[~data2.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data1_1 = data1_1.drop("level_1", axis=1)
data2_1 = data2_1.drop("level_1", axis=1)
n1 = len(data1_1["StockID"].unique())
n2 = len(data2_1["StockID"].unique())
print("The number of SH level2 stocks in baseline broker is: " + str(n1))
print("The number of SH level2 stocks in new broker is: " + str(n2))
# merge and compare
columns = ["StockID", "cum_volume", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
data1_1 = data1_1[data1_1["StockID"].isin(data2_1["StockID"].unique())]
test = pd.merge(data1_1, data2_1, left_on=columns, right_on=columns, how="outer")
n1 = test["sequenceNo_x"].count()
n2 = test["sequenceNo_y"].count()
len1 = len(test)
d1 = 0
m1 = 0
d2 = 0
m2 = 0
m = []
n = 0
for i in data2_1["StockID"].unique():
    if len(set(data1[data1["StockID"] == i]["cum_volume"].unique()) - set(data2[data2["StockID"] == i]["cum_volume"].unique())) == 0:
        n = n + 1
    else:
        m.append(i)

if (n2 < len1) & (n1 < len1):
    print("1. New Borker has different lv2 data compared with baseline broker:")
    print("The number of different data:")
    print(len1 - n2 + len1 - n1)
    print("The number of shared data:")
    print(n2 + n1 - len1)
    m1 = sum(np.isnan(pd.merge(test[np.isnan(test["sequenceNo_x"])].loc[:, ["StockID", "time_y", "sequenceNo_y"]],
                               data1, left_on=["StockID", "time_y"], right_on=["StockID", "time"], how="left")
                      ["sequenceNo"]))
    d1 = len1 - n1 - m1
    m2 = sum(np.isnan(pd.merge(test[np.isnan(test["sequenceNo_y"])].loc[:, ["StockID", "time_x", "sequenceNo_x"]],
                               data2, left_on=["StockID", "time_x"], right_on=["StockID", "time"], how="left")
                      ["sequenceNo"]))
    d2 = len1 - n2 - m2
    print("(1) Two brokers have different data: " + str(d1 + d2))
    print("(2) Baseline broker miss some data: " + str(m1))
    print("(3) New broker miss some data: " + str(m2))
    data1[(data1["StockID"] == 600000) & (data1["time"] >= 135102000) & (data1["time"] <= 135108000)].loc[:,
    ["source", "StockID", "time", "cum_volume", "cum_amount", "close", "ask1p", "ask1q"]].astype(object)

# 2.3 index data
print("\nstart to compare index data")
data1 = logSH[logSH["StockID"].isin(in_dex) & (logSH["cum_volume"] > 0) & (logSH["time"] <= 145700000)]
data2 = logSH1[logSH1["StockID"].isin(in_dex) & (logSH1["cum_volume"] > 0) & (logSH1["time"] <= 145700000)]
data1["cum_amount"] = data1["cum_amount"].apply(lambda x: c_round(x, 0))
columns = ["cum_volume", "cum_amount", "close", "openPrice"]
# drop duplicates
data1_1 = data1.groupby("StockID")[data1.columns[~data1.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data2_1 = data2.groupby("StockID")[data2.columns[~data2.columns.isin(["StockID"])].values].apply(
    lambda x: x[~x.duplicated(columns, keep="first")]).reset_index()
data1_1 = data1_1.drop("level_1", axis=1)
data2_1 = data2_1.drop("level_1", axis=1)
# merge and compare
columns = ["StockID", "cum_volume", "close", "openPrice"]
test = pd.merge(data1_1, data2_1, left_on=columns, right_on=columns, how="outer")
n1 = test["sequenceNo_x"].count()
n2 = test["sequenceNo_y"].count()
len1 = len(test)
d1 = 0
m1 = 0
d2 = 0
m2 = 0
if (n2 < len1) & (n1 < len1):
    print("1. New Borker has different index data compared with baseline broker:")
    print("The number of different data:")
    print(len1 - n2 + len1 - n1)
    print("The number of shared data:")
    print(n2 + n1 - len1)
    print("2. The reason of difference:")
    m1 = sum(np.isnan(pd.merge(test[np.isnan(test["sequenceNo_x"])].loc[:, ["StockID", "time_y", "sequenceNo_y"]],
                               data1, left_on=["StockID", "time_y"], right_on=["StockID", "time"], how="left")
                               ["sequenceNo"]))
    d1 = len1 - n1 - m1
    m2 = sum(np.isnan(pd.merge(test[np.isnan(test["sequenceNo_y"])].loc[:, ["StockID", "time_x", "sequenceNo_x"]],
                               data1, left_on=["StockID", "time_x"], right_on=["StockID", "time"], how="left")
                               ["sequenceNo"]))
    d2 = len1 - n2 - m2

    print("(1) Two brokers have different data: " + str(d1 + d2))
    print("(2) Baseline broker miss some data: " + str(m1))
    print("(3) New broker miss some data: " + str(m2))