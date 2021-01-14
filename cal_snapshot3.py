import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter
import collections
import bisect

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

# df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
# df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
# df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
# df2 = df2.sort_values("sequenceNo")
# df3 = df3[df3["exchId"] == 2]
# df3 = df3.sort_values("sequenceNo")
# print("get data now")
# F1 = open("E:\\forZhenyu\\logs_20200120_zs_92_01_day_data\\mdLog_SZ_20200120_0843.pkl",'rb')
# F2 = open("C:\\Users\\win\\Downloads\\forZhenyu\\orderLog.pkl",'rb')
# F3 = open("C:\\Users\\win\\Downloads\\forZhenyu\\tradeLog.pkl",'rb')
# df1 = pickle.load(F1).iloc[:, 1:]
# df2 = pickle.load(F2).iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
# df3 = pickle.load(F3).iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
df1 = pd.read_csv("E:\\mbd\\raw data\\mdLog_SZ_20200207_0838.csv", encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv("E:\\mbd\\raw data\\mdOrderLog_20200207_0838.csv", encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pd.read_csv("E:\\mbd\\raw data\\mdTradeLog_20200207_0838.csv", encoding="utf-8").iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]

df2 = df2.sort_values("sequenceNo")
df3 = df3[df3["exchId"] == 2]
df3 = df3.sort_values("sequenceNo")
print("get data now")

def create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i):
    ob_dict1 = collections.OrderedDict(sorted(ob_dict1.items()))
    ob_dict2 = collections.OrderedDict(sorted(ob_dict2.items()))
    if db[i]["TransactTime"] >= 93000000:
        if len([k for k, v in ob_dict2.items() if v != 0]) == 0:
            m_in = 99999999
        else:
            m_in = min(k for k, v in ob_dict2.items() if (v != 0) & (k != 0))
        if len([k for k, v in ob_dict1.items() if v != 0]) == 0:
            m_ax = 0
        else:
            m_ax = max(k for k, v in ob_dict1.items() if v != 0)

        if m_ax < m_in:
            if db[i]["TransactTime"] < open_time:
                op = 0
            else:
                op = op1
            qty = round(qty, 2)
            len1 = len([k for k, v in ob_dict1.items() if v != 0])
            len2 = len([k for k, v in ob_dict2.items() if v != 0])
            l1 = [k for k, v in ob_dict1.items() if v != 0]
            v1 = [v for k, v in ob_dict1.items() if v != 0]
            l2 = [k for k, v in ob_dict2.items() if (v != 0) & (k != 0)]
            v2 = [v for k, v in ob_dict2.items() if (v != 0) & (k != 0)]
            if len1 < 5:
                for i1 in range(len1, 5):
                    l1.insert(0, 0)
                    v1.insert(0, 0)
            else:
                l1 = l1[-5:]
                v1 = v1[-5:]

            if len2 < 5:
                for i2 in range(len2, 5):
                    l2.append(0)
                    v2.append(0)
            else:
                l2 = l2[:5]
                v2 = v2[:5]
            [b5, b4, b3, b2, b1] = [x / 10000 for x in l1]
            [bv5, bv4, bv3, bv2, bv1] = v1
            [a1, a2, a3, a4, a5] = [x / 10000 for x in l2]
            [av1, av2, av3, av4, av5] = v2
            dic[i] = [db[i]["sequenceNo"], 100, db["SecurityID"][0], "SZ", db[i]["TransactTime"], vol, qty, cl,
                      b1, b2, b3, b4, b5, bv1, bv2, bv3, bv4, bv5, a1, a2, a3, a4, a5, av1, av2, av3, av4, av5, op]


def snapshot(ID):
    test1 = df2[df2["SecurityID"] == ID]
    test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
    trade1 = df3[df3["SecurityID"] == ID]
    re1 = df1[df1["StockID"] == ID]

    dic = dict()
    cancel_dict = dict()
    price_dict = dict()
    ob_dict1 = dict()
    ob_dict2 = dict()
    extra_dict1 = dict()
    extra_dict2 = dict()
    db = pd.concat([test1, trade1]).sort_values(by=["sequenceNo"])
    db = db[
        ["sequenceNo", "exchId", "SecurityID", "TransactTime", "ApplSeqNum", "Side", "Price", "OrderType", "OrderQty",
         "ExecType", "TradePrice", "TradeQty", "TradeMoney", "BidApplSeqNum", "OfferApplSeqNum"]]

    # during auction & normal time
    db = db[db["TransactTime"] <= 145655000]
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
            if db[i]["TransactTime"] < 93000000:
                if s == 1:
                    if p in ob_dict1.keys():
                        ob_dict1[p] = ob_dict1[p] + q
                    else:
                        ob_dict1.update({p: q})
                if s == 2:
                    if p in ob_dict2.keys():
                        ob_dict2[p] = ob_dict2[p] + q
                    else:
                        ob_dict2.update({p: q})
            else:
                if db[i - 1]["ExecType"] == "F":
                    if (1, db[i - 1]["BidApplSeqNum"]) in cancel_dict.keys():
                        if cancel_dict[1, db[i - 1]["BidApplSeqNum"]] != 0:
                            p = db[i - 1]["TradePrice"]
                            price_dict[db[i - 1]["BidApplSeqNum"]] = [p, 1, "1"]
                            if p in ob_dict1.keys():
                                ob_dict1[p] = ob_dict1[p] + cancel_dict[1, db[i - 1]["BidApplSeqNum"]]
                            else:
                                ob_dict1.update({p: cancel_dict[1, db[i - 1]["BidApplSeqNum"]]})
                            create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i - 1)
                            del cancel_dict[1, db[i - 1]["BidApplSeqNum"]]
                        else:
                            del cancel_dict[1, db[i - 1]["BidApplSeqNum"]]

                    if (2, db[i - 1]["OfferApplSeqNum"]) in cancel_dict.keys():
                        if cancel_dict[2, db[i - 1]["OfferApplSeqNum"]] != 0:
                            p = db[i - 1]["TradePrice"]
                            price_dict[db[i - 1]["OfferApplSeqNum"]] = [p, 2, "1"]
                            if p in ob_dict2.keys():
                                ob_dict2[p] = ob_dict2[p] + cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]
                            else:
                                ob_dict2.update({p: cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]})
                            create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i - 1)
                            del cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]
                        else:
                            del cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]

                p = db[i]["Price"]
                if s == 1:
                    l = [k for k, v in ob_dict2.items() if v != 0]
                    if len(l) == 0:
                        if p in ob_dict1.keys():
                            ob_dict1[p] = ob_dict1[p] + q
                        else:
                            ob_dict1.update({p: q})
                    else:
                        if p < min(l):
                            if p in ob_dict1.keys():
                                ob_dict1[p] = ob_dict1[p] + q
                            else:
                                ob_dict1.update({p: q})
                        else:
                            l = sorted(l)
                            m1 = bisect.bisect_left(l, p)
                            bisect.insort_left(l, p)
                            for m in l[:(m1 + 1)]:
                                if m in [k for k, v in ob_dict2.items() if v != 0]:
                                    q1 = ob_dict2[m]
                                    if q > q1:
                                        if m in ob_dict1.keys():
                                            # ob_dict1[m] = ob_dict1[m] + q1
                                            ob_dict2[m] = ob_dict2[m] - q1
                                            vol = vol + q1
                                            qty = qty + q1 * m / 10000
                                            q = q - q1
                                            cl = m / 10000
                                            extra_dict1.update({(db[i]["ApplSeqNum"], m): q1})
                                            if (m == p) & (q != 0):
                                                ob_dict1[m] = ob_dict1[m] + q
                                        else:
                                            # ob_dict1.update({m: q1})
                                            ob_dict2[m] = ob_dict2[m] - q1
                                            vol = vol + q1
                                            qty = qty + q1 * m / 10000
                                            q = q - q1
                                            cl = m / 10000
                                            extra_dict1.update({(db[i]["ApplSeqNum"], m): q1})
                                            if (m == p) & (q != 0):
                                                ob_dict1.update({m: q})
                                    else:
                                        if m in ob_dict1.keys():
                                            # ob_dict1[m] = ob_dict1[m] + q
                                            ob_dict2[m] = ob_dict2[m] - q
                                            vol = vol + q
                                            qty = qty + q * m / 10000
                                            cl = m / 10000
                                            extra_dict1.update({(db[i]["ApplSeqNum"], m): q})
                                            break
                                        else:
                                            # ob_dict1.update({m: q})
                                            ob_dict2[m] = ob_dict2[m] - q
                                            vol = vol + q
                                            qty = qty + q * m / 10000
                                            cl = m / 10000
                                            extra_dict1.update({(db[i]["ApplSeqNum"], m): q})
                                            break
                                else:
                                    if m in ob_dict1.keys():
                                        ob_dict1[m] = ob_dict1[m] + q
                                    else:
                                        ob_dict1.update({m: q})

                else:
                    l = [k for k, v in ob_dict1.items() if v != 0]
                    if len(l) == 0:
                        if p in ob_dict2.keys():
                            ob_dict2[p] = ob_dict2[p] + q
                        else:
                            ob_dict2.update({p: q})
                    else:
                        if p > max(l):
                            if p in ob_dict2.keys():
                                ob_dict2[p] = ob_dict2[p] + q
                            else:
                                ob_dict2.update({p: q})
                        else:
                            l = sorted(l)
                            m2 = bisect.bisect_right(l, p)
                            bisect.insort_left(l, p)
                            for m in l[m2:][::-1]:
                                if m in [k for k, v in ob_dict1.items() if v != 0]:
                                    q1 = ob_dict1[m]
                                    if q > q1:
                                        if m in ob_dict2.keys():
                                            # ob_dict2[m] = ob_dict2[m] + q1
                                            ob_dict1[m] = ob_dict1[m] - q1
                                            vol = vol + q1
                                            qty = qty + q1 * m / 10000
                                            q = q - q1
                                            cl = m / 10000
                                            extra_dict2.update({(db[i]["ApplSeqNum"], m): q1})
                                            if (m == p) & (q != 0):
                                                ob_dict2[m] = ob_dict2[m] + q
                                        else:
                                            # ob_dict2.update({m: q1})
                                            ob_dict1[m] = ob_dict1[m] - q1
                                            vol = vol + q1
                                            qty = qty + q1 * m / 10000
                                            q = q - q1
                                            cl = m / 10000
                                            extra_dict2.update({(db[i]["ApplSeqNum"], m): q1})
                                            if (m == p) & (q != 0):
                                                ob_dict2.update({m: q})
                                    else:
                                        if m in ob_dict2.keys():
                                            ob_dict1[m] = ob_dict1[m] - q
                                            vol = vol + q
                                            qty = qty + q * m / 10000
                                            cl = m / 10000
                                            extra_dict2.update({(db[i]["ApplSeqNum"], m): q})
                                            # ob_dict2[m] = ob_dict2[m] + q
                                            break
                                        else:
                                            ob_dict1[m] = ob_dict1[m] - q
                                            vol = vol + q
                                            qty = qty + q * m / 10000
                                            cl = m / 10000
                                            extra_dict2.update({(db[i]["ApplSeqNum"], m): q})
                                            # ob_dict2.update({m: q})
                                            break
                                else:
                                    if m in ob_dict2.keys():
                                        ob_dict2[m] = ob_dict2[m] + q
                                    else:
                                        ob_dict2.update({m: q})

        elif db[i]["OrderType"] == str(1):
            cancel_dict[s, db[i]["ApplSeqNum"]] = q


        elif db[i]["OrderType"] == "U":
            if s == 1:
                if len([k for k, v in ob_dict1.items() if v != 0]) == 0:
                    p = cl * 10000
                else:
                    p = max(k for k, v in ob_dict1.items() if v != 0)
                ob_dict1[p] = ob_dict1[p] + q
                price_dict[db[i]["ApplSeqNum"]] = [p, s, db[i]["OrderType"]]
            if s == 2:
                if len([k for k, v in ob_dict2.items() if v != 0]) == 0:
                    p = cl * 10000
                else:
                    p = min(k for k, v in ob_dict2.items() if v != 0)
                ob_dict2[p] = ob_dict2[p] + q
                price_dict[db[i]["ApplSeqNum"]] = [p, s, db[i]["OrderType"]]

        else:
            if db[i]["ExecType"] == str(4):
                if db[i - 1]["ExecType"] == "F":
                    if (1, db[i - 1]["BidApplSeqNum"]) in cancel_dict.keys():
                        if cancel_dict[1, db[i - 1]["BidApplSeqNum"]] != 0:
                            p = db[i - 1]["TradePrice"]
                            price_dict[db[i - 1]["BidApplSeqNum"]] = [p, 1, "1"]
                            if p in ob_dict1.keys():
                                ob_dict1[p] = ob_dict1[p] + cancel_dict[1, db[i - 1]["BidApplSeqNum"]]
                            else:
                                ob_dict1.update({p: cancel_dict[1, db[i - 1]["BidApplSeqNum"]]})
                            create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i - 1)
                            del cancel_dict[1, db[i - 1]["BidApplSeqNum"]]
                        else:
                            del cancel_dict[1, db[i - 1]["BidApplSeqNum"]]
                    if (2, db[i - 1]["OfferApplSeqNum"]) in cancel_dict.keys():
                        if cancel_dict[2, db[i - 1]["OfferApplSeqNum"]] != 0:
                            p = db[i - 1]["TradePrice"]
                            price_dict[db[i - 1]["OfferApplSeqNum"]] = [p, 2, "1"]
                            if p in ob_dict2.keys():
                                ob_dict2[p] = ob_dict2[p] + cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]
                            else:
                                ob_dict2.update({p: cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]})
                            create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i - 1)
                            del cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]
                        else:
                            del cancel_dict[2, db[i - 1]["OfferApplSeqNum"]]
                if db[i - 1]["OrderType"] == "1":
                    continue
                num = db[i]["BidApplSeqNum"] + db[i]["OfferApplSeqNum"]
                if num not in price_dict.keys():
                    print(num)
                line = price_dict[num]
                pr = line[0]
                di = line[1]
                if di == 1:
                    ob_dict1[pr] = ob_dict1[pr] - db[i]["TradeQty"]
                else:
                    ob_dict2[pr] = ob_dict2[pr] - db[i]["TradeQty"]
            if db[i]["ExecType"] == "F":
                n1 = db[i]["BidApplSeqNum"]
                n2 = db[i]["OfferApplSeqNum"]
                pr = db[i]["TradePrice"]
                if (n1, pr) in extra_dict1.keys():
                    extra_dict1[n1, pr] = extra_dict1[n1, pr] - db[i]["TradeQty"]
                    if extra_dict1[n1, pr] == 0:
                        del extra_dict1[n1, pr]
                    continue

                if (n2, pr) in extra_dict2.keys():
                    extra_dict2[n2, pr] = extra_dict2[n2, pr] - db[i]["TradeQty"]
                    if extra_dict2[n2, pr] == 0:
                        del extra_dict2[n2, pr]
                    continue

                vol = vol + db[i]["TradeQty"]
                qty = qty + db[i]["TradeMoney"] / 10000
                cl = db[i]["TradePrice"] / 10000
                if db[i]["TransactTime"] < 93000000:
                    pr1 = price_dict[n1][0]
                    pr2 = price_dict[n2][0]
                    ob_dict1[pr1] = ob_dict1[pr1] - db[i]["TradeQty"]
                    ob_dict2[pr2] = ob_dict2[pr2] - db[i]["TradeQty"]
                else:
                    if (1, n1) in cancel_dict.keys():
                        ob_dict2[pr] = ob_dict2[pr] - db[i]["TradeQty"]
                        cancel_dict[1, n1] = cancel_dict[1, n1] - db[i]["TradeQty"]
                    elif (2, n2) in cancel_dict.keys():
                        ob_dict1[pr] = ob_dict1[pr] - db[i]["TradeQty"]
                        cancel_dict[2, n2] = cancel_dict[2, n2] - db[i]["TradeQty"]
                    else:
                        ob_dict1[pr] = ob_dict1[pr] - db[i]["TradeQty"]
                        ob_dict2[pr] = ob_dict2[pr] - db[i]["TradeQty"]



      # create snapshot data
        create_snapshot(ob_dict1, ob_dict2, db, open_time, op1, qty, vol, cl, dic, i)

    myre = pd.DataFrame.from_dict(dic, orient='index', columns=["sequenceNo", "source", "ID", "exchange",
                                                                "time", "cum_volume", "cum_amount", "close", "bid1p",
                                                                "bid2p", "bid3p", "bid4p", "bid5p",
                                                                "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p",
                                                                "ask2p", "ask3p", "ask4p", "ask5p",
                                                                "ask1q", "ask2q", "ask3q", "ask4q", "ask5q",
                                                                "openPrice"]).reset_index().iloc[:, 1:]
    ID = 2000000 + ID
    myre.to_csv("E:\\mbd\\mbd data\\Tick_" + str(ID) + ".csv", encoding="utf-8")
    print(str(ID) + " collected")
#     re1 = pd.read_csv("E:\\forZhenyu\\v2\\Tick_" + str(ID) + ".csv", encoding="utf-8")
#     print("start to compare")
#     # re1["time"] = re1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
# #    num = list(set(re1[(re1["time"] >= 93003000) & (re1["time"] <= 112957000) & (re1["source"] == 4)].index) |
# #            set(re1[(re1["time"] >= 130003000) & (re1["time"] <= 145657000) & (re1["source"] == 4)].index))
#     num = list(set(re1[(re1["time"] >= 93000000) & (re1["time"] < 113000000)].index) |
#                set(re1[(re1["time"] >= 130000000) & (re1["time"] <= 145657000)].index))
#     num.sort()
#     columns = ["ID", "cum_volume", "cum_amount", "sequenceNo", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
#                "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
#                "ask2q", "ask3q", "ask4q", "ask5q"]
#     c = 0
#     len3 = len(num)
#     myre["ID"] = 2000000 + myre["ID"]
#     for n1 in num:
#         if myre[myre["cum_volume"] == re1.loc[n1, "cum_volume"]].loc[:, columns].append(
#                 re1.loc[n1, columns]).duplicated().iloc[-1]:
#             c = c + 1
#     print("The total number of level2 snapshot data of stock_" + str(ID) + " is: " + str(len3))
#     print("The number of matched snapshot data of stock_" + str(ID) + " is: " + str(c))

# for i in np.sort(df2["SecurityID"].unique()):
for i in np.sort(df2["SecurityID"].unique()):
    snapshot(i)

# test1 = df2[df2["SecurityID"] == 662]
# test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
# trade1 = df3[df3["SecurityID"] == 662]
# re1 = df1[df1["StockID"] == 662]
# myre = pd.read_csv("E:\\Snapshot data1\\" + str(662) + ".csv", encoding="utf-8")
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
# print("The total number of level2 snapshot data of stock_" + str(662) + " is: " + str(len3))
# print("The number of matched snapshot data of stock_" + str(2356) + " is: " + str(c))