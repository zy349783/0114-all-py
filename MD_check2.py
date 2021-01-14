import pandas as pd
import random
import numpy as np

TradeLog = pd.read_csv('E:\\compare\\58\\mdTradeLog_20200107_0845.csv',
                       encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "exchId", "TransactTime",
                                                 "ApplSeqNum", "SecurityID", "ExecType", "TradeBSFlag",
                                                 "TradePrice", "TradeQty", "TradeMoney", "BidApplSeqNum",
                                                 "OfferApplSeqNum"]]
# OrderLog = pd.read_csv('E:\\compare\\oes\\mdOrderLog_20200106_0912.csv',
#                        encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "exchId", "TransactTime",
#                                                  "ApplSeqNum", "SecurityID", "Side", "OrderType", "Price",
#                                                  "OrderQty"]]
TradeLog1 = pd.read_csv('E:\\compare\\zt_58_0107\\Logs\\mdTradeLog_20200107_0858.csv',
                        encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "exchId", "TransactTime",
                                                  "ApplSeqNum", "SecurityID", "ExecType", "TradeBSFlag",
                                                  "TradePrice", "TradeQty", "TradeMoney", "BidApplSeqNum",
                                                  "OfferApplSeqNum", "ChannelNo"]]
# OrderLog1 = pd.read_csv('E:\\compare\\52_zs_0106\\Logs\\mdOrderLog_20200106_0910.csv',
#                         encoding="utf-8").loc[:, ["clockAtArrival", "sequenceNo", "exchId", "TransactTime",
#                                                   "ApplSeqNum", "SecurityID", "Side", "OrderType", "Price",
#                                                   "OrderQty", "ChannelNo"]]

# SZE Trade + Order Data completeness check
OrderLog["OrderType"] = OrderLog["OrderType"].apply(lambda x: str(x))
OrderLog1["OrderType"] = OrderLog1["OrderType"].apply(lambda x: str(x))
TradeLogSZ = TradeLog[TradeLog["exchId"] == 2]
TradeLogSZ1 = TradeLog1[TradeLog1["exchId"] == 2]
stocks = random.choices(OrderLog1["SecurityID"].unique(), k=50)
OrderLogSZ = OrderLog[OrderLog["SecurityID"].isin(stocks)]
OrderLogSZ1 = OrderLog1[OrderLog1["SecurityID"].isin(stocks)]
TradeLogSZ = TradeLogSZ[TradeLogSZ["SecurityID"].isin(stocks)]
TradeLogSZ1 = TradeLogSZ1[TradeLogSZ1["SecurityID"].isin(stocks)]

stocks = random.choices(OrderLog1["SecurityID"].unique(), k=50)
OrderLogSZ = OrderLog[OrderLog["SecurityID"].isin(stocks)]
del OrderLog
OrderLogSZ1 = OrderLog1[OrderLog1["SecurityID"].isin(stocks)]
del OrderLog1
TradeLogSZ = TradeLog[TradeLog["SecurityID"].isin(stocks)]
TradeLogSZ1 = TradeLog1[TradeLog1["SecurityID"].isin(stocks)]


TradeLogSZ1_1 = TradeLogSZ1[TradeLogSZ1["ChannelNo"] == -1]
OrderLogSZ1_1 = OrderLogSZ1[OrderLogSZ1["ChannelNo"] == -1]
TradeLogSZ1_2 = TradeLogSZ1[TradeLogSZ1["ChannelNo"] != -1]
OrderLogSZ1_2 = OrderLogSZ1[OrderLogSZ1["ChannelNo"] != -1]
SZ = pd.concat([TradeLogSZ, OrderLogSZ]).sort_values(by=["sequenceNo"])
SZ1 = pd.concat([TradeLogSZ1_1, OrderLogSZ1_1]).sort_values(by=["sequenceNo"])
SZ2 = pd.concat([TradeLogSZ1_2, OrderLogSZ1_2]).sort_values(by=["sequenceNo"])
SZ1["TradeMoney"] = SZ1["TradeMoney"] * 10000
SZ1["OrderType"] = SZ1["OrderType"].apply(lambda x: str(x))
SZ["OrderType"] = SZ["OrderType"].apply(lambda x: str(x))
SZ2["OrderType"] = SZ2["OrderType"].apply(lambda x: str(x))

columns = ["TransactTime", "ApplSeqNum", "SecurityID", "TradePrice", "TradeQty",
           "BidApplSeqNum", "OfferApplSeqNum"]
columns = ["TransactTime", "ApplSeqNum", "SecurityID", "Side", "OrderType", "Price", "OrderQty"]
# columns = ["TransactTime", "ApplSeqNum", "SecurityID", "ExecType", "TradeBSFlag", "TradePrice", "TradeQty",
#            "TradeMoney", "BidApplSeqNum", "OfferApplSeqNum", "Side", "OrderType", "Price", "OrderQty"]
columns = ["TransactTime", "ApplSeqNum", "SecurityID", "TradePrice", "TradeQty",
           "BidApplSeqNum", "OfferApplSeqNum", "Side", "OrderType", "Price", "OrderQty"]
re = pd.merge(SZ, SZ1, left_on=columns, right_on=columns, how="outer")
n1 = re["sequenceNo_x"].count()
n2 = re["sequenceNo_y"].count()
len1 = len(re)

re[np.isnan(re["sequenceNo_y"])]


# SZE Trade + Order Data ordering within stocks check
stocks = SZ["SecurityID"].unique()
m = 0
diff = []
for i in stocks:
    re1 = pd.merge(SZ[SZ["SecurityID"] == i], SZ1[SZ1["SecurityID"] == i], left_on=columns, right_on=columns, how="outer")
    if (sorted(re1["sequenceNo_y"]) != re1["sequenceNo_y"]).any() == False:
        m = m + 1
    else:
        diff.append(i)
if m == len(stocks):
    print("SZE Trade + Order Data have same ordering within stocks")

# SZE Trade + Order Data ordering among stocks check
if (sorted(re["sequenceNo_y"]) != re["sequenceNo_y"]).any():
    print("SZE Trade + Order Data don't have same ordering among stocks")






# SHE Trade Data completeness check
SH = TradeLog[TradeLog["exchId"] == 1]
SH1 = TradeLog1[TradeLog1["exchId"] == 1]
SH["cum_volume"] = SH.groupby("SecurityID")["TradeQty"].cumsum()
SH["cum_amount"] = SH.groupby("SecurityID")["TradeMoney"].cumsum()
stocks = SH1["SecurityID"].unique()
SH = SH[SH["SecurityID"].isin(stocks)]
columns = ["TransactTime", "ApplSeqNum", "SecurityID", "TradePrice", "TradeQty", "TradeMoney", "TradeBSFlag",
           "BidApplSeqNum", "OfferApplSeqNum"]
re = pd.merge(SH, SH1, left_on=columns, right_on=columns, how="outer")
n1 = re["sequenceNo_x"].count()
n2 = re["sequenceNo_y"].count()
len1 = len(re)

# SHE Trade Data ordering within stocks check
stocks = SH["SecurityID"].unique()
m = 0
diff = []
for i in stocks:
    re1 = pd.merge(SH[SH["SecurityID"] == i], SH1[SH1["SecurityID"] == i], left_on=columns, right_on=columns, how="outer")
    if (sorted(re1["sequenceNo_y"]) != re1["sequenceNo_y"]).any() == False:
        m = m + 1
    else:
        diff.append(i)
if m == len(stocks):
    print("SHE Trade Data have same ordering within stocks")
else:
    print("SHE Trade Data don't have same ordering within stocks")


# SZE Trade + Order Data ordering among stocks check
if (sorted(re["sequenceNo_y"]) != re["sequenceNo_y"]).any():
    print("SHE Trade Data don't have same ordering among stocks")
else:
    print("SHE Trade Data have same ordering among stocks")