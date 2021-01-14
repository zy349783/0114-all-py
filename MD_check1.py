import pandas as pd
import numpy as np

logSZ = pd.read_csv('E:\\compare\\oes\\mdLog_SZ_20200106_0912.csv',
                    encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "source", "StockID",
                                              "exchange", "time", "cum_volume", "cum_amount", "close",
                                              "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q",
                                              "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                              "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
                                              "ask4q", "ask5q", "openPrice"]]
logSZ1 = pd.read_csv('E:\\compare\\52_zs_0106\\Logs\\mdLog_SZ_20200106_0910.csv',
                     encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "source", "StockID",
                                               "exchange", "time", "cum_volume", "cum_amount", "close",
                                               "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q",
                                               "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                               "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
                                               "ask4q", "ask5q", "openPrice"]]
# 1. data format
# 1.1 in general
print("original data source type: ")
print(logSZ["source"].unique())
print("new data source type: ")
print(logSZ1["source"].unique())
logSZ["time"] = logSZ["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
logSZ["cum_volume"] = logSZ["cum_volume"].round(0)
list = logSZ1[logSZ1["time"].str.len() == 8].index.values
list1 = logSZ1[logSZ1["time"].str.len() != 8].index.values
logSZ1.loc[list, "time"] = logSZ1.loc[list, "time"].apply(lambda x: int((x.replace(':', "")).ljust(9, "0")))
logSZ1.loc[list1, "time"] = logSZ1.loc[list1, "time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
# 1.2 level1 & level2 data ("price", "TransactTime")
print(logSZ[(logSZ["StockID"] == 1) & (logSZ["time"] > 93000000) & (logSZ["source"] == 3)].head())
print(logSZ1[(logSZ1["StockID"] == 1) & (logSZ1["time"] > 93000000) & (logSZ1["source"] == 24)].head())
print(logSZ[(logSZ["StockID"] == 1) & (logSZ["time"] > 93000000) & (logSZ["source"] == 4)].head())


# 2. data accuracy
# 2.1 level1 data
print("start to compare level1 data")
data1 = logSZ[(logSZ["cum_volume"] > 0) & (logSZ["time"] < 145700000) & (logSZ["source"] == 3)]
data2 = logSZ1[(logSZ1["cum_volume"] > 0) & (logSZ1["time"] < 145700000) & (logSZ1["source"] == 20)]
columns = ["cum_volume", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
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
print("The number of SZ level1 stocks in baseline broker is: " + str(n1))
print("The number of SZ level1 stocks in new broker is: " + str(n2))
# merge and compare
columns = ["StockID", "cum_volume", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
data2_1 = data2_1[data2_1["StockID"].isin(data1_1["StockID"].unique())]
test = pd.merge(data1_1, data2_1, left_on=columns, right_on=columns, how="outer")
n1 = test["sequenceNo_x"].count()
n2 = test["sequenceNo_y"].count()
len1 = len(test)
if n2 == len1:
    print("1. New Broker has same lv1 data as baseline broker")
    if n1 == n2:
        print("baseline broker has same lv1 data as new broker")




# 2.2 level2 data
print("\nstart to compare level2 data")
data1 = logSZ[(logSZ["cum_volume"] > 0) & (logSZ["time"] < 145700000) & (logSZ["source"] == 4)]
data2 = logSZ1[(logSZ1["cum_volume"] > 0) & (logSZ1["time"] < 145700000) & (logSZ1["source"] == 21)]
columns = ["cum_volume", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
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
print("The number of SZ level2 stocks in baseline broker is: " + str(n1))
print("The number of SZ level2 stocks in new broker is: " + str(n2))
# merge and compare
columns = ["StockID", "cum_volume", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p", "bid1q", "bid2q",
           "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q",
           "ask4q", "ask5q", "openPrice"]
data1_1 = data1_1[data1_1["StockID"].isin(data2_1["StockID"].unique())]
test = pd.merge(data1_1, data2_1, left_on=columns, right_on=columns, how="outer")
n1 = test["sequenceNo_x"].count()
n2 = test["sequenceNo_y"].count()
len1 = len(test)
if n2 == len1:
    print("New Broker has same lv2 data as baseline broker")
    if n1 < n2:
        print("Baseline broker has smaller lv2 data compared with new broker:")
        print("Stocks that differs:")
        print(test[np.isnan(test["sequenceNo_x"])]["StockID"].unique())
    if n1 == n2:
        print("baseline broker has same lv2 data as new broker")

