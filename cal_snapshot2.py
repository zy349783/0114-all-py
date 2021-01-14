import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter
import collections

def to_tensor(dataframe, columns = [], dtypes = {}, index = False):
    to_records_kwargs = {'index': index}
    if not columns:  # Default to all `dataframe.columns`
        columns = dataframe.columns
    if dtypes:       # Pull in modifications only for dtypes listed in `columns`
        to_records_kwargs['column_dtypes'] = {}
        for column in dtypes.keys():
            if column in columns:
                to_records_kwargs['column_dtypes'].update({column: dtypes.get(column)})
    return dataframe[columns].to_records(**to_records_kwargs)

df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
df2 = df2.sort_values("sequenceNo")
df3 = df3[df3["exchId"] == 2]
df3 = df3.sort_values("sequenceNo")
print("get data now")
# F1 = open("C:\\Users\\win\\Downloads\\forZhenyu\\mdLog.pkl",'rb')
# F2 = open("C:\\Users\\win\\Downloads\\forZhenyu\\orderLog.pkl",'rb')
# F3 = open("C:\\Users\\win\\Downloads\\forZhenyu\\tradeLog.pkl",'rb')
# df1 = pickle.load(F1).iloc[:, 1:]
# df2 = pickle.load(F2).iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
# df3 = pickle.load(F3).iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
# df2 = df2.sort_values("sequenceNo")
# df3 = df3[df3["exchId"] == 2]
# df3 = df3.sort_values("sequenceNo")
# print("get data now")


def snapshot(ID):
    test1 = df2[df2["SecurityID"] == ID]
    test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
    trade1 = df3[df3["SecurityID"] == ID]
    re1 = df1[df1["StockID"] == ID]

    myre = pd.DataFrame()
    dic = dict()
    cancel_dict = dict()
    price_dict = dict()
    ob_dict = dict()
    db = pd.concat([test1, trade1]).sort_values(by=["sequenceNo"])
    db = db[
        ["sequenceNo", "exchId", "SecurityID", "TransactTime", "ApplSeqNum", "Side", "Price", "OrderType", "OrderQty",
         "ExecType", "TradePrice", "TradeQty", "TradeMoney", "BidApplSeqNum", "OfferApplSeqNum"]]

    # during auction & normal time
    db = to_tensor(db)
    if len(db[(db["TransactTime"] < 93000000) & (db["ExecType"] == "F")]["TradePrice"]) != 0:
        op1 = db[(db["TransactTime"] < 93000000) & (db["ExecType"] == "F")]["TradePrice"][-1] / 10000
        open_time = 93000000
    else:
        op1 = db[(db["TransactTime"] >= 93000000) & (db["ExecType"] == "F")]["TradePrice"][0] / 10000
        open_time = db[(db["TransactTime"] >= 93000000) & (db["ExecType"] == "F")]["TransactTime"][0]
    vol = 0
    qty = 0
    cl = 0
    for i in range(len(db)):
        s = db[i]["Side"]
        p = db[i]["Price"]
        q = db[i]["OrderQty"]
        if db[i]["OrderType"] == str(2):
            price_dict[db[i]["ApplSeqNum"]] = [p, s, db[i]["OrderType"]]
            if (s, p) in ob_dict.keys():
                ob_dict[s, p] = ob_dict[s, p] + q
            else:
                ob_dict.update({(s, p): q})

        elif db[i]["OrderType"] == str(1):
            price_dict[db[i]["ApplSeqNum"]] = [0, s, db[i]["OrderType"]]
            if s == 1:
                td = db[db["BidApplSeqNum"] == db[i]["ApplSeqNum"]]
                if (td["ExecType"] == str(4)).any():
                    td[-1]["TradePrice"] = td["TradePrice"][0]
                    cancel_dict[db[i]["ApplSeqNum"]] = td["TradePrice"][0]
                if td["TradeQty"].sum() < q:
                    ob_dict[1, td["TradePrice"][0]] = ob_dict[1, td["TradePrice"][0]] + q - td["TradeQty"].sum()
                for j in range(len(td)):
                    if (1, td[j]["TradePrice"]) in ob_dict.keys():
                        ob_dict[1, td[j]["TradePrice"]] = ob_dict[1, td[j]["TradePrice"]] + td[j]["TradeQty"]
                    else:
                        ob_dict.update({(1, td[j]["TradePrice"]): td[j]["TradeQty"]})
            if s == 2:
                td = db[db["OfferApplSeqNum"] == db[i]["ApplSeqNum"]]
                if (td["ExecType"] == str(4)).any():
                    td[-1]["TradePrice"] = td["TradePrice"][0]
                    cancel_dict[db[i]["ApplSeqNum"]] = td["TradePrice"][0]
                if td["TradeQty"].sum() < q:
                    ob_dict[2, td["TradePrice"][0]] = ob_dict[2, td["TradePrice"][0]] + q - td["TradeQty"].sum()
                for j in range(len(td)):
                    if (2, td[j]["TradePrice"]) in ob_dict.keys():
                        ob_dict[2, td[j]["TradePrice"]] = ob_dict[2, td[j]["TradePrice"]] + td[j]["TradeQty"]
                    else:
                        ob_dict.update({(2, td[j]["TradePrice"]): td[j]["TradeQty"]})

        elif db[i]["OrderType"] == "U":
            if s == 1:
                if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]) == 0:
                    if db[db["BidApplSeqNum"] == db[i]["ApplSeqNum"]]["ExecType"][0] == str(4):
                        p = cl * 10000
                    else:
                        p = db[db["BidApplSeqNum"] == db[i]["ApplSeqNum"]]["TradePrice"][0]
                else:
                    p = max(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1))[1]
                ob_dict[s, p] = ob_dict[s, p] + q
                price_dict[db[i]["ApplSeqNum"]] = [p, s, db[i]["OrderType"]]
            if s == 2:
                if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)]) == 0:
                    if db[db["OfferApplSeqNum"] == db[i]["ApplSeqNum"]]["ExecType"][0] == str(4):
                        p = cl * 10000
                    else:
                        p = db[db["OfferApplSeqNum"] == db[i]["ApplSeqNum"]]["TradePrice"][0]
                else:
                    p = min(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2))[1]
                ob_dict[s, p] = ob_dict[s, p] + q
                price_dict[db[i]["ApplSeqNum"]] = [p, s, db[i]["OrderType"]]

        else:
            if db[i]["ExecType"] == str(4):
                num = db[i]["BidApplSeqNum"] + db[i]["OfferApplSeqNum"]
                line = price_dict[num]
                if line[2] == str(1):
                    pr = cancel_dict[num]
                else:
                    pr = line[0]
                di = line[1]
                ob_dict[di, pr] = ob_dict[di, pr] - db[i]["TradeQty"]
            if db[i]["ExecType"] == "F":
                n1 = db[i]["BidApplSeqNum"]
                n2 = db[i]["OfferApplSeqNum"]
                pr1 = price_dict[n1][0]
                pr2 = price_dict[n2][0]
                if pr1 == 0:
                    pr1 = db[i]["TradePrice"]
                if pr2 == 0:
                    pr2 = db[i]["TradePrice"]
                ob_dict[1, pr1] = ob_dict[1, pr1] - db[i]["TradeQty"]
                ob_dict[2, pr2] = ob_dict[2, pr2] - db[i]["TradeQty"]
                vol = vol + db[i]["TradeQty"]
                qty = qty + db[i]["TradeMoney"] / 10000
                cl = db[i]["TradePrice"] / 10000

      # create snapshot data
        ob_dict = collections.OrderedDict(sorted(ob_dict.items()))
        if db[i]["TransactTime"] >= 93000000:
            if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)]) == 0:
                m_in = 99999999
            else:
                m_in = min(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2) & (k[1] != 0))[1]
            if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]) == 0:
                m_ax = 0
            else:
                m_ax = max(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1))[1]

            if m_ax < m_in:
                if db[i]["TransactTime"] < open_time:
                    op = 0
                else:
                    op = op1
                qty = round(qty, 2)
                len1 = len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)])
                len2 = len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)])
                l1 = [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]
                v1 = [ob_dict[x] for x in [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]]
                l2 = [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2) & (k[1] != 0)]
                v2 = [ob_dict[x] for x in [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2) & (k[1] != 0)]]
                if len1 < 5:
                    for i1 in range(len1, 5):
                        l1.insert(0, (1, 0))
                        v1.insert(0, 0)
                else:
                    l1 = l1[-5:]
                    v1 = v1[-5:]

                if len2 < 5:
                    for i2 in range(len2, 5):
                        l2.append((2, 0))
                        v2.append(0)
                else:
                    l2 = l2[:5]
                    v2 = v2[:5]
                [b5, b4, b3, b2, b1] = [x[1] / 10000 for x in l1]
                [bv5, bv4, bv3, bv2, bv1] = v1
                [a1, a2, a3, a4, a5] = [x[1] / 10000 for x in l2]
                [av1, av2, av3, av4, av5] = v2
                dic[i] = [db[i]["sequenceNo"], 100, db["SecurityID"][0], "SZ", db[i]["TransactTime"], vol, qty, cl,
                          b1, b2, b3, b4, b5, bv1, bv2, bv3, bv4, bv5, a1, a2, a3, a4, a5, av1, av2, av3, av4, av5, op]


    myre = pd.DataFrame.from_dict(dic, orient='index', columns=["sequenceNo", "source", "StockID", "exchange",
                                                                "time", "cum_volume", "cum_amount", "close", "bid1p",
                                                                "bid2p", "bid3p", "bid4p", "bid5p",
                                                                "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p",
                                                                "ask2p", "ask3p", "ask4p", "ask5p",
                                                                "ask1q", "ask2q", "ask3q", "ask4q", "ask5q",
                                                                "openPrice"]).reset_index().iloc[:, 1:]
    myre.to_csv("E:\\Snapshot data\\" + str(ID) + ".csv", encoding="utf-8")
    print(str(ID) + " collected")
    # myre = pd.read_csv("E:\\Snapshot data\\" + str(2442) + ".csv", encoding="utf-8")
    print("start to compare")
    re1["time"] = re1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
    num = list(set(re1[(re1["time"] >= 93003000) & (re1["time"] <= 112957000) & (re1["source"] == 4)].index) |
               set(re1[(re1["time"] >= 130003000) & (re1["time"] <= 145657000) & (re1["source"] == 4)].index))
    num.sort()
    columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
               "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
               "ask2q", "ask3q", "ask4q", "ask5q", "openPrice"]
    c = 0
    len3 = len(num)
    for n1 in num:
        if myre[myre["cum_volume"] == re1.loc[n1, "cum_volume"]].loc[:, columns].append(
                re1.loc[n1, columns]).duplicated().iloc[-1]:
            c = c + 1
    print("The total number of level2 snapshot data of stock_" + str(ID) + " is: " + str(len3))
    print("The number of matched snapshot data of stock_" + str(ID) + " is: " + str(c))

# for i in df2["SecurityID"].unique()[588:]:
#     snapshot(i)

snapshot(662)

# test1 = df2[df2["SecurityID"] == 2356]
# test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
# trade1 = df3[df3["SecurityID"] == 2356]
# re1 = df1[df1["StockID"] == 2356]
# myre = pd.read_csv("E:\\Snapshot data result\\" + str(2356) + ".csv", encoding="utf-8")
# print("start to compare")
# re1["time"] = re1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
# num = list(set(re1[(re1["time"] >= 93003000) & (re1["time"] <= 112957000) & (re1["source"] == 4)].index) |
#                set(re1[(re1["time"] >= 130003000) & (re1["time"] <= 145657000) & (re1["source"] == 4)].index))
# num.sort()
# columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
#                "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
#                "ask2q", "ask3q", "ask4q", "ask5q", "openPrice"]
# c = 0
# len3 = len(num)
# for n1 in num:
#     if myre[myre["cum_volume"] == re1.loc[n1, "cum_volume"]].loc[:, columns].append(
#                 re1.loc[n1, columns]).duplicated().iloc[-1]:
#         c = c + 1
#     else:
#         print(myre[myre["cum_volume"] == re1.loc[n1, "cum_volume"]].loc[:, columns].append(
#                 re1.loc[n1, columns]))
# print("The total number of level2 snapshot data of stock_" + str(2356) + " is: " + str(len3))
# print("The number of matched snapshot data of stock_" + str(2356) + " is: " + str(c))
# #
#
#
#
