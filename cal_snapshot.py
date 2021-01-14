import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter

df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 5, 8, 9, 12, 13, 14, 15, 16]]
df2 = df2.sort_values("sequenceNo")
df3 = df3.sort_values("sequenceNo")
print("get data now")
test1 = df2[df2["SecurityID"] == 631]
test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
trade1 = df3[df3["SecurityID"] == 631]
re1 = df1[df1["StockID"] == 631]
# test1 = df2[df2["SecurityID"] == 166]
# trade1 = df3[df3["SecurityID"] == 166]
# re1 = df1[df1["StockID"] == 166]
# test3 = df2[df2["SecurityID"] == 2442]
# trade3 = df3[df3["SecurityID"] == 2442]
# re3 = df1[df1["StockID"] == 2442]

myre = pd.DataFrame(columns=["sequenceNo", "source", "StockID", "exchange", "time", "cum_volume",
                             "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
                             "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                             "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q", "ask4q",
                             "ask5q", "openPrice"])
db = pd.concat([test1, trade1]).sort_values(by=["sequenceNo"])
db = db[["sequenceNo", "exchId", "SecurityID", "TransactTime", "ApplSeqNum", "Side", "Price", "OrderType", "OrderQty",
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
        db1.loc[db1["ApplSeqNum"] == trade_bid[i], "OrderQty"] = db1.loc[db1["ApplSeqNum"] == trade_bid[i], "OrderQty"] \
                                                                 - db2[db2["ExecType"] == "F"]["TradeQty"].values[i]
    trade_ask = db2[db2["ExecType"] == "F"]["OfferApplSeqNum"].values
    for i in range(len(trade_ask)):
        db1.loc[db1["ApplSeqNum"] == trade_ask[i], "OrderQty"] = db1.loc[db1["ApplSeqNum"] == trade_ask[i], "OrderQty"] \
                                                                 - db2[db2["ExecType"] == "F"]["TradeQty"].values[i]

o_b = pd.merge(db1.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
                 db1.groupby(["Side", "Price"]).last().reset_index()[db1.columns[~db1.columns.isin(["OrderQty"])]],
                 left_on=["Side", "Price"], right_on=["Side", "Price"])

# during normal time
for i in db[db["TransactTime"] >= 93000000].index:
    if db.loc[i, "OrderType"] == str(2):
        o_b = o_b.append(db.loc[i, :])
        o_b = pd.merge(o_b.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
              o_b.groupby(["Side", "Price"]).last().reset_index()[o_b.columns[~o_b.columns.isin(["OrderQty"])]],
              left_on=["Side", "Price"], right_on=["Side", "Price"])
    if db.loc[i, "OrderType"] == str(1):
        dup = db.loc[i, :]
        if dup.loc["Side"] == 1:
            dup.loc["Price"] = myre.tail(1)["ask1p"].values[0]
            o_b = o_b.append(dup)
            o_b = pd.merge(o_b.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
                           o_b.groupby(["Side", "Price"]).last().reset_index()[o_b.columns[~o_b.columns.isin(
                           ["OrderQty"])]], left_on=["Side", "Price"], right_on=["Side", "Price"])
        if dup.loc["Side"] == 2:
            dup.loc["Price"] = myre.tail(1)["bid1p"].values[0]
            o_b = o_b.append(dup)
            o_b = pd.merge(o_b.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
                           o_b.groupby(["Side", "Price"]).last().reset_index()[o_b.columns[~o_b.columns.isin(
                               ["OrderQty"])]], left_on=["Side", "Price"], right_on=["Side", "Price"])
    if db.loc[i, "OrderType"] == "U":
        dup = db.loc[i, :]
        if dup.loc["Side"] == 1:
            dup.loc["Price"] = myre.tail(1)["bid1p"].values[0]
            o_b = o_b.append(dup)
            o_b = pd.merge(o_b.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
                           o_b.groupby(["Side", "Price"]).last().reset_index()[o_b.columns[~o_b.columns.isin(
                           ["OrderQty"])]], left_on=["Side", "Price"], right_on=["Side", "Price"])
        if dup.loc["Side"] == 2:
            dup.loc["Price"] = myre.tail(1)["ask1p"].values[0]
            o_b = o_b.append(dup)
            o_b = pd.merge(o_b.groupby(["Side", "Price"]).sum().reset_index().loc[:, ["Side", "Price", "OrderQty"]],
                           o_b.groupby(["Side", "Price"]).last().reset_index()[o_b.columns[~o_b.columns.isin(
                               ["OrderQty"])]], left_on=["Side", "Price"], right_on=["Side", "Price"])
    else:
        if db.loc[i, "ExecType"] == str(4):
            num = np.sum(db.loc[i, ["BidApplSeqNum", "OfferApplSeqNum"]])
            pr = db.loc[db["ApplSeqNum"] == num, "Price"].values[0]
            di = db.loc[db["ApplSeqNum"] == num, "Side"].values[0]
            o_b.loc[(o_b["Side"] == di) & (o_b["Price"] == pr), "OrderQty"] = o_b.loc[(o_b["Side"] == di) &
                    (o_b["Price"] == pr), "OrderQty"] - db.loc[i, "TradeQty"]
        if db.loc[i, "ExecType"] == "F":
            n1 = db.loc[i, "BidApplSeqNum"]
            n2 = db.loc[i, "OfferApplSeqNum"]
            pr1 = db.loc[db["ApplSeqNum"] == n1, "Price"].values[0]
            pr2 = db.loc[db["ApplSeqNum"] == n2, "Price"].values[0]
            o_b.loc[(o_b["Side"] == 1) & (o_b["Price"] == pr1), "OrderQty"] = o_b.loc[(o_b["Side"] == 1) &
                    (o_b["Price"] == pr1), "OrderQty"] - db.loc[i, "TradeQty"]
            o_b.loc[(o_b["Side"] == 2) & (o_b["Price"] == pr2), "OrderQty"] = o_b.loc[(o_b["Side"] == 2) &
                    (o_b["Price"] == pr2), "OrderQty"] - db.loc[i, "TradeQty"]
# create Snapshot data under given time
    vol = trade1[(trade1["sequenceNo"] <= db.loc[i, "sequenceNo"]) & (trade1["ExecType"] == "F")]["TradeQty"].sum()
    qty = trade1[(trade1["sequenceNo"] <= db.loc[i, "sequenceNo"]) & (trade1["ExecType"] == "F")]["TradeMoney"].sum() / 10000
    op = trade1[(trade1["TransactTime"] <= 93000000) & (trade1["ExecType"] != str(4))]["TradePrice"].tail(1).values[
             0] / 10000
    b1 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "Price"].iloc[-1] / 10000
    b2 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "Price"].iloc[-2] / 10000
    b3 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "Price"].iloc[-3] / 10000
    b4 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "Price"].iloc[-4] / 10000
    b5 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "Price"].iloc[-5] / 10000
    bv1 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[-1]
    bv2 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[-2]
    bv3 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[-3]
    bv4 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[-4]
    bv5 = o_b.loc[(o_b["Side"] == 1) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[-5]
    a1 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "Price"].iloc[0] / 10000
    a2 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "Price"].iloc[1] / 10000
    a3 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "Price"].iloc[2] / 10000
    a4 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "Price"].iloc[3] / 10000
    a5 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "Price"].iloc[4] / 10000
    av1 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[0]
    av2 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[1]
    av3 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[2]
    av4 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[3]
    av5 = o_b.loc[(o_b["Side"] == 2) & (o_b["OrderQty"] != 0), "OrderQty"].iloc[4]
    cl = trade1[(trade1["sequenceNo"] <= db.loc[i, "sequenceNo"]) & (trade1["ExecType"] != str(4))]\
             ["TradePrice"].tail(1).values[0] / 10000
    t = db.loc[i, "TransactTime"]
    myre = myre.append({"sequenceNo": db.loc[i, "sequenceNo"], "source": 100, "StockID": test1["SecurityID"].iloc[0],
                        "exchange": "SZ", "time": t, "cum_volume": vol, "cum_amount": qty, "close": cl, "bid1p": b1,
                        "bid2p": b2, "bid3p": b3, "bid4p": b4, "bid5p": b5, "bid1q": bv1, "bid2q": bv2, "bid3q": bv3,
                        "bid4q": bv4, "bid5q": bv5, "ask1p": a1, "ask2p": a2, "ask3p": a3, "ask4p": a4, "ask5p": a5,
                        "ask1q": av1, "ask2q": av2, "ask3q": av3, "ask4q": av4, "ask5q": av5, "openPrice": op},
                         ignore_index=True)

myre = myre[["sequenceNo", "source", "StockID", "exchange", "time", "cum_volume",
                 "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
                 "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                 "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q", "ask4q",
                 "ask5q", "openPrice"]]





















li_st = sorted(list(set(test1["sequenceNo"].values) | set(trade1["sequenceNo"].values)))
for i in li_st:
    if i in test1["sequenceNo"].values:
        if test1.loc[test1["sequenceNo"] == i, "Side"].values[0] == 1:
            if test1.loc[test1["sequenceNo"] == i, "OrderType"].values[0] == 2:
                if test1.loc[test1["sequenceNo"] == i, "Price"].values in order_book["buy"].values:
                    order_book.loc[
                        order_book["buy"] == test1.loc[test1["sequenceNo"] == i, "Price"].values[0], "buy_vol"] = \
                        order_book.loc[
                            order_book["buy"] == test1.loc[test1["sequenceNo"] == i, "Price"].values[0], "buy_vol"] + \
                            test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0]
                else:
                    order_book = order_book.append({"buy": test1.loc[test1["sequenceNo"] == i, "Price"].values[0],
                                                    "buy_vol": test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[
                                                        0]}, ignore_index=True)
            # if test1.loc[test1["sequenceNo"] == i, "OrderType"].values[0] == 1:
            #     if test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] <= \
            #             myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask1q"].values[0]:
            #         order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         ,"ask1p"].values[0], "buy_vol"] = order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "ask1p"].values[0], "buy_vol"] + test1.loc[
            #         test1["sequenceNo"] == i, "OrderQty"].values[0]
            #     if (test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] >
            #         myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask1q"].values[0]) & \
            #             (test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] <=
            #              myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask1q"].values[0] +
            #              myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask2q"].values[0]):
            #         order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         , "ask1p"].values[0], "buy_vol"] = order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "ask1p"].values[0], "buy_vol"] + myre.loc[
            #         myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask1q"].values[0]
            #         order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         , "ask2p"].values[0], "buy_vol"] = order_book.loc[order_book["buy"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "ask2p"].values[0], "buy_vol"] + test1.loc[
            #         test1["sequenceNo"] == i, "OrderQty"].values[0] - myre.loc[
            #         myre["sequenceNo"] == li_st[li_st.index(i) - 1], "ask1q"].values[0]

        if test1.loc[test1["sequenceNo"] == i, "Side"].values[0] == 2:
            if test1.loc[test1["sequenceNo"] == i, "OrderType"].values[0] == 2:
                if test1.loc[test1["sequenceNo"] == i, "Price"].values in order_book["sell"].values:
                    order_book.loc[
                        order_book["sell"] == test1.loc[test1["sequenceNo"] == i, "Price"].values[0], "sell_vol"] = \
                        order_book.loc[
                            order_book["sell"] == test1.loc[test1["sequenceNo"] == i, "Price"].values[0], "sell_vol"] + \
                            test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0]
                else:
                    order_book = order_book.append({"sell": test1.loc[test1["sequenceNo"] == i, "Price"].values[0],
                                                    "sell_vol": test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[
                                                        0]}, ignore_index=True)
            # if test1.loc[test1["sequenceNo"] == i, "OrderType"].values[0] == 1:
            #     if test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] <= \
            #             myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid1q"].values[0]:
            #         order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         ,"bid1p"].values[0], "sell_vol"] = order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "bid1p"].values[0], "sell_vol"] + test1.loc[
            #         test1["sequenceNo"] == i, "OrderQty"].values[0]
            #     if (test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] >
            #         myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid1q"].values[0]) & \
            #             (test1.loc[test1["sequenceNo"] == i, "OrderQty"].values[0] <=
            #              myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid1q"].values[0] +
            #              myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid2q"].values[0]):
            #         order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         , "bid1p"].values[0], "sell_vol"] = order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "bid1p"].values[0], "sell_vol"] + myre.loc[
            #         myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid1q"].values[0]
            #         order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"] == li_st[li_st.index(i) - 1]
            #         , "bid2p"].values[0], "sell_vol"] = order_book.loc[order_book["sell"] == myre.loc[myre["sequenceNo"]
            #         == li_st[li_st.index(i) - 1], "bid2p"].values[0], "sell_vol"] + test1.loc[
            #         test1["sequenceNo"] == i, "OrderQty"].values[0] - myre.loc[
            #         myre["sequenceNo"] == li_st[li_st.index(i) - 1], "bid1q"].values[0]


    else:
        if trade1.loc[trade1["sequenceNo"] == i, "ExecType"].values[0] == str(4):
            num = (trade1.loc[trade1["sequenceNo"] == i, "BidApplSeqNum"] +
                   trade1.loc[trade1["sequenceNo"] == i, "OfferApplSeqNum"]).values[0]
            d_el = test1[test1["ApplSeqNum"] == num]
            if d_el["Side"].values[0] == 1:
                order_book.loc[order_book["buy"] == d_el["Price"].values[0], "buy_vol"] = \
                order_book.loc[order_book["buy"] == d_el["Price"].values[0], "buy_vol"] - \
                trade1.loc[trade1["sequenceNo"] == i, "TradeQty"].values[0]
            else:
                order_book.loc[order_book["sell"] == d_el["Price"].values[0], "sell_vol"] = \
                order_book.loc[order_book["sell"] == d_el["Price"].values[0], "sell_vol"] - \
                trade1.loc[trade1["sequenceNo"] == i, "TradeQty"].values[0]
        if trade1.loc[trade1["sequenceNo"] == i, "ExecType"].values[0] == "F":
            order_book.loc[order_book["buy"] == test1.loc[
                test1["ApplSeqNum"] == trade1.loc[trade1["sequenceNo"] == i, "BidApplSeqNum"].values[0],
                "Price"].values[0], "buy_vol"] = order_book.loc[order_book["buy"] == test1.loc[
                test1["ApplSeqNum"] == trade1.loc[trade1["sequenceNo"] == i, "BidApplSeqNum"].values[0],
                "Price"].values[0], "buy_vol"] - trade1.loc[trade1["sequenceNo"] == i, "TradeQty"].values[0]
            order_book.loc[order_book["sell"] == test1.loc[
                test1["ApplSeqNum"] == trade1.loc[trade1["sequenceNo"] == i, "OfferApplSeqNum"].values[0],
                "Price"].values[0], "sell_vol"] = order_book.loc[order_book["sell"] == test1.loc[
                test1["ApplSeqNum"] == trade1.loc[trade1["sequenceNo"] == i, "OfferApplSeqNum"].values[0],
                "Price"].values[0], "sell_vol"] - trade1.loc[trade1["sequenceNo"] == i, "TradeQty"].values[0]

    if i >= test1[test1["TransactTime"] >= 93000000]["sequenceNo"].head(1).values[0]:
        order_book = order_book.sort_values(by=["sell", "buy"], ascending=False)
        vol = trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] == "F")]["TradeQty"].sum()
        qty = (trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] == "F")]["TradeQty"] *
               trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] == "F")]["TradePrice"]).sum() / 10000
        op = trade1[(trade1["TransactTime"] <= 93000000) & (trade1["ExecType"] != str(4))]["TradePrice"].tail(1).values[
                 0] / 10000
        b1 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy"].iloc[0] / 10000
        b2 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy"].iloc[1] / 10000
        b3 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy"].iloc[2] / 10000
        b4 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy"].iloc[3] / 10000
        b5 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy"].iloc[4] / 10000
        bv1 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy_vol"].iloc[0]
        bv2 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy_vol"].iloc[1]
        bv3 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy_vol"].iloc[2]
        bv4 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy_vol"].iloc[3]
        bv5 = order_book[(order_book["sell"].isna()) & (order_book["buy_vol"] != 0)]["buy_vol"].iloc[4]
        a1 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell"].iloc[-1] / 10000
        a2 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell"].iloc[-2] / 10000
        a3 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell"].iloc[-3] / 10000
        a4 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell"].iloc[-4] / 10000
        a5 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell"].iloc[-5] / 10000
        av1 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell_vol"].iloc[-1]
        av2 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell_vol"].iloc[-2]
        av3 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell_vol"].iloc[-3]
        av4 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell_vol"].iloc[-4]
        av5 = order_book[(order_book["buy"].isna()) & (order_book["sell_vol"] != 0)]["sell_vol"].iloc[-5]
        cl = trade1[(trade1["sequenceNo"] <= i) & (trade1["ExecType"] != str(4))]["TradePrice"].tail(1).values[
                 0] / 10000
        if i in test1["sequenceNo"].values:
            t = test1.loc[test1["sequenceNo"] == i, "TransactTime"].values[0]
        else:
            t = trade1.loc[trade1["sequenceNo"] == i, "TransactTime"].values[0]
        myre = myre.append({"sequenceNo": i, "source": 100, "StockID": test1["SecurityID"].iloc[0], "exchange": "SZ",
                            "time": t, "cum_volume": vol, "cum_amount": qty, "close": cl, "bid1p": b1, "bid2p": b2,
                            "bid3p": b3, "bid4p": b4, "bid5p": b5, "bid1q": bv1, "bid2q": bv2, "bid3q": bv3,
                            "bid4q": bv4, "bid5q": bv5, "ask1p": a1, "ask2p": a2, "ask3p": a3, "ask4p": a4,
                            "ask5p": a5, "ask1q": av1, "ask2q": av2, "ask3q": av3, "ask4q": av4, "ask5q": av5,
                            "openPrice": op}, ignore_index=True)
        myre = myre[["sequenceNo", "source", "StockID", "exchange", "time", "cum_volume",
                                     "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
                                     "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p",
                                     "ask3p", "ask4p", "ask5p", "ask1q", "ask2q", "ask3q", "ask4q",
                                     "ask5q", "openPrice"]]

print("start to compare")
diff = pd.DataFrame()
re1["time"] = re1["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
num = list(set(re1[(re1["time"] >= 93003000) & (re1["time"] <= 112957000) & (re1["source"] == 4)].index) |
           set(re1[(re1["time"] >= 130003000) & (re1["time"] <= 145657000) & (re1["source"] == 4)].index))
num.sort()
for n1 in num:
    pair = pd.Index([i for i in re1.loc[n1, "time"] - myre["time"]]).get_loc(
        min(i for i in re1.loc[n1, "time"] - myre["time"] if i > 0))
    columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
               "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
               "ask2q", "ask3q", "ask4q", "ask5q", "openPrice"]
    if isinstance(pair, int):
        df = pd.concat([myre.loc[pair, columns], re1.loc[n1, columns]], axis=1).T.diff().iloc[1, :]
        df["time1"] = re1.loc[n1, "time"]
        df["time2"] = myre.loc[pair, "time"]
    else:
        df = pd.concat([myre.loc[pair, columns].tail(1).T, re1.loc[n1, columns]], axis=1).T.diff().iloc[1, :]
        df = df.append(pd.Series(re1.loc[n1, "time"], index=["time1"]))
        df = df.append(pd.Series(myre.loc[pair, "time"].tail(1).values[0], index=["time2"]))

    diff = pd.concat([diff, df], axis=1)

d_f = diff.loc[:, (diff.iloc[:25, :] != 0).any()].T
n = len(diff.loc["time1", :][(diff.iloc[:25, :] != 0).any()])
d_f.to_csv("E:\\Snapshot data\\" + str(631) + "_diff.csv", encoding="utf-8")
print("The number of unmatched level2 snapshot data of stocks "+ str(631) + " is " + str(n))
diff = diff.T
print(d_f)
print(myre)
print(df1.columns)