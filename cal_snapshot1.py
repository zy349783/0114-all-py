import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter
import collections

df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
df2 = df2.sort_values("sequenceNo")
df3 = df3.sort_values("sequenceNo")
print("get data now")

def snapshot(ID):
    test1 = df2[df2["SecurityID"] == ID]
    test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
    trade1 = df3[df3["SecurityID"] == ID]
    re1 = df1[df1["StockID"] == ID]

    myre = pd.DataFrame(columns=["sequenceNo", "source", "StockID", "exchange", "time", "cum_volume",
                                 "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
                                 "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                 "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q", "ask4q",
                                 "ask5q", "openPrice"])
    db = pd.concat([test1, trade1]).sort_values(by=["sequenceNo"])
    db = db[
        ["sequenceNo", "exchId", "SecurityID", "TransactTime", "ApplSeqNum", "Side", "Price", "OrderType", "OrderQty",
         "ExecType", "TradePrice", "TradeQty", "TradeMoney", "BidApplSeqNum", "OfferApplSeqNum"]]

    # during auction time
    db1 = db[db["TransactTime"] < 93000000]
    db2 = db1[db1["OrderType"].isnull()]
    if (db2["ExecType"] == str(4)).any():
        d_el = np.sum(db2.loc[db2["ExecType"] == str(4), ["BidApplSeqNum", "OfferApplSeqNum"]], axis=1).values
        for i in range(len(d_el)):
            db1.loc[db1["ApplSeqNum"] == d_el[i], "OrderQty"] = db1.loc[db1["ApplSeqNum"] == d_el[i], "OrderQty"] - \
                                                                db2[db2["ExecType"] == str(4)]["TradeQty"].values[i]

    if (db2["ExecType"] == "F").any():
        trade_bid = db2[db2["ExecType"] == "F"]["BidApplSeqNum"].values
        for i in range(len(trade_bid)):
            db1.loc[db1["ApplSeqNum"] == trade_bid[i], "OrderQty"] = db1.loc[
                                                                         db1["ApplSeqNum"] == trade_bid[i], "OrderQty"] \
                                                                     - db2[db2["ExecType"] == "F"]["TradeQty"].values[i]
        trade_ask = db2[db2["ExecType"] == "F"]["OfferApplSeqNum"].values
        for i in range(len(trade_ask)):
            db1.loc[db1["ApplSeqNum"] == trade_ask[i], "OrderQty"] = db1.loc[
                                                                         db1["ApplSeqNum"] == trade_ask[i], "OrderQty"] \
                                                                     - db2[db2["ExecType"] == "F"]["TradeQty"].values[i]
    o_b = db1.groupby(["Side", "Price"]).sum().reset_index()
    o_b = o_b.set_index(["Side", "Price"])
    ob_dict = o_b["OrderQty"].to_dict()

    # during normal time
    for i in db.loc[db["TransactTime"] >= 93000000, "sequenceNo"].values:
        s = db.loc[db["sequenceNo"] == i, "Side"].values[0]
        p = db.loc[db["sequenceNo"] == i, "Price"].values[0]
        q = db.loc[db["sequenceNo"] == i, "OrderQty"].values[0]
        if db.loc[db["sequenceNo"] == i, "OrderType"].values[0] == str(2):
            if (s, p) in ob_dict.keys():
                ob_dict[s, p] = ob_dict[s, p] + q
            else:
                ob_dict.update({(s, p): q})

        elif db.loc[db["sequenceNo"] == i, "OrderType"].values[0] == str(1):
            db.loc[db["sequenceNo"] == i, "Price"] = 0
            if s == 1:
                td = trade1[trade1["BidApplSeqNum"] == db.loc[db["sequenceNo"] == i, "ApplSeqNum"].values[0]]. \
                    groupby("TradePrice")["TradeQty"].sum().reset_index()
                for j in range(len(td)):
                    if (1, td.iloc[j, 0]) in ob_dict.keys():
                        ob_dict[1, td.iloc[j, 0]] = ob_dict[1, td.iloc[j, 0]] + td.iloc[j, 1]
                    else:
                        ob_dict.update({(1, td.iloc[j, 0]): td.iloc[j, 1]})
            if s == 2:
                td = trade1[trade1["OfferApplSeqNum"] == db.loc[db["sequenceNo"] == i, "ApplSeqNum"].values[0]]. \
                    groupby("TradePrice")["TradeQty"].sum().reset_index()
                for j in range(len(td)):
                    if (2, td.iloc[j, 0]) in ob_dict.keys():
                        ob_dict[2, td.iloc[j, 0]] = ob_dict[2, td.iloc[j, 0]] + td.iloc[j, 1]
                    else:
                        ob_dict.update({(2, td.iloc[j, 0]): td.iloc[j, 1]})

        elif db.loc[db["sequenceNo"] == i, "OrderType"].values[0] == "U":
            db.loc[db["sequenceNo"] == i, "Price"] = 0
            if s == 1:
                p = max(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1))[1]
                ob_dict[s, p] = ob_dict[s, p] + q
            if s == 2:
                p = min(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2))[1]
                ob_dict[s, p] = ob_dict[s, p] + q


        else:
            if db.loc[db["sequenceNo"] == i, "ExecType"].values[0] == str(4):
                num = (trade1.loc[trade1["sequenceNo"] == i, "BidApplSeqNum"] +
                       trade1.loc[trade1["sequenceNo"] == i, "OfferApplSeqNum"]).values[0]
                pr = db.loc[db["ApplSeqNum"] == num, "Price"].values[0]
                di = db.loc[db["ApplSeqNum"] == num, "Side"].values[0]
                ob_dict[di, pr] = ob_dict[di, pr] - db.loc[db["sequenceNo"] == i, "TradeQty"].values[0]
            if db.loc[db["sequenceNo"] == i, "ExecType"].values[0] == "F":
                n1 = db.loc[db["sequenceNo"] == i, "BidApplSeqNum"].values[0]
                n2 = db.loc[db["sequenceNo"] == i, "OfferApplSeqNum"].values[0]
                pr1 = db.loc[db["ApplSeqNum"] == n1, "Price"].values[0]
                pr2 = db.loc[db["ApplSeqNum"] == n2, "Price"].values[0]
                if pr1 == 0:
                    pr1 = db.loc[db["sequenceNo"] == i, "TradePrice"].values[0]
                if pr2 == 0:
                    pr2 = db.loc[db["sequenceNo"] == i, "TradePrice"].values[0]
                ob_dict[1, pr1] = ob_dict[1, pr1] - db.loc[db["sequenceNo"] == i, "TradeQty"].values[0]
                ob_dict[2, pr2] = ob_dict[2, pr2] - db.loc[db["sequenceNo"] == i, "TradeQty"].values[0]

        # create Snapshot data under given time
        ob_dict = collections.OrderedDict(sorted(ob_dict.items()))
        if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)]) == 0:
            m_in = 99999999
        else:
            m_in = min(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2))[1]
        if len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]) == 0:
            m_ax = 0
        else:
            m_ax = max(k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1))[1]
        if m_ax < m_in:
            vol = trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] == "F")]["TradeQty"].sum()
            qty = trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] == "F")]["TradeMoney"].sum() / 10000
            op = \
            trade1[(trade1["TransactTime"] <= 93000000) & (trade1["ExecType"] != str(4))]["TradePrice"].tail(1).values[
                0] / 10000
            len1 = len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)])
            len2 = len([k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)])
            l1 = [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]
            v1 = [ob_dict[x] for x in [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 1)]]
            l2 = [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)]
            v2 = [ob_dict[x] for x in [k for k, v in ob_dict.items() if (v != 0) & (k[0] == 2)]]
            print(i)
            if len1 < 5:
                for i1 in range(len1, 5):
                    l1.append((1, 0))
                    v1.append(0)
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
            print(l1)
            print(l2)
            print(i)
            [b5, b4, b3, b2, b1] = [x[1] / 10000 for x in l1]
            [bv5, bv4, bv3, bv2, bv1] = v1
            [a1, a2, a3, a4, a5] = [x[1] / 10000 for x in l2]
            [av1, av2, av3, av4, av5] = v2
            cl = trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] != str(4))]["TradePrice"].tail(1).values[
                     0] / 10000
            t = db.loc[db["sequenceNo"] == i, "TransactTime"].values[0]

            myre = myre.append({"sequenceNo": i, "source": 100, "StockID": test1["SecurityID"].iloc[0],
                                "exchange": "SZ", "time": t, "cum_volume": vol, "cum_amount": qty, "close": cl,
                                "bid1p": b1,
                                "bid2p": b2, "bid3p": b3, "bid4p": b4, "bid5p": b5, "bid1q": bv1, "bid2q": bv2,
                                "bid3q": bv3,
                                "bid4q": bv4, "bid5q": bv5, "ask1p": a1, "ask2p": a2, "ask3p": a3, "ask4p": a4,
                                "ask5p": a5,
                                "ask1q": av1, "ask2q": av2, "ask3q": av3, "ask4q": av4, "ask5q": av5, "openPrice": op},
                               ignore_index=True)
    myre = myre[["sequenceNo", "source", "StockID", "exchange", "time", "cum_volume",
                 "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
                 "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                 "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q", "ask4q",
                 "ask5q", "openPrice"]]

    # compare the result
    myre.to_csv("E:\\Snapshot data\\" + str(ID) + ".csv", encoding="utf-8")
    # myre = pd.read_csv("E:\\Snapshot data\\" + str(2442) + ".csv", encoding="utf-8")
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
        else:
            print(myre[myre["cum_volume"] == re1.loc[n1, "cum_volume"]].loc[:, columns].append(re1.loc[n1, columns]))
    print("The total number of level2 snapshot data of stock_" + str(ID) + " is: " + str(len3))
    print("The number of matched snapshot data of stock_" + str(ID) + " is: " + str(c))

for i in df2["SecurityID"].unique():
    snapshot(i)







# diff = pd.DataFrame()
# re1["time"] = re1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
# num = list(set(re1[(re1["time"] >= 93003000) & (re1["time"] <= 112957000) & (re1["source"] == 4)].index) |
#            set(re1[(re1["time"] >= 130003000) & (re1["time"] <= 145657000) & (re1["source"] == 4)].index))
# num.sort()
# for n1 in num:
#     pair = pd.Index([i for i in re1.loc[n1, "time"] - myre["time"]]).get_loc(
#         min(i for i in re1.loc[n1, "time"] - myre["time"] if i >= 0))
#     columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
#                "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
#                "ask2q", "ask3q", "ask4q", "ask5q", "openPrice"]
#     if isinstance(pair, int):
#         df = pd.concat([myre.loc[pair, columns], re1.loc[n1, columns]], axis=1).T.diff().iloc[1, :]
#         df["time1"] = re1.loc[n1, "time"]
#         df["time2"] = myre.loc[pair, "time"]
#     else:
#         df = pd.concat([myre.loc[pair, columns].tail(1).T, re1.loc[n1, columns]], axis=1).T.diff().iloc[1, :]
#         df = df.append(pd.Series(re1.loc[n1, "time"], index=["time1"]))
#         df = df.append(pd.Series(myre.loc[pair, "time"].tail(1).values[0], index=["time2"]))
#
#     diff = pd.concat([diff, df], axis=1)
#
# d_f = diff.loc[:, (diff.iloc[:25, :] != 0).any()].T
# n = len(diff.loc["time1", :][(diff.iloc[:25, :] != 0).any()])
# print(d_f)
# d_f.to_csv("E:\\Snapshot data\\" + str(166) + "_diff.csv", encoding="utf-8")
# print("The number of unmatched level2 snapshot data of stocks " + str(631) + " is " + str(n))
# diff = diff.T
# print(d_f)
# print(myre)
# print(df1.columns)