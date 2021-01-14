import pandas as pd
import matplotlib as mpl
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import timedelta
import calendar
import seaborn as sns

# 1. calculate beta with respect to SH000905
# sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
# in_dex = pd.read_csv('D:\\work\\project 3 event study\\index_daily_SH000300.csv',
#                            encoding="utf-8").iloc[:, 1:]
# sto_ck["ret"] = sto_ck["ret"]/100
# in_dex["returns"] = in_dex['close']/in_dex['close'].shift(1) - 1
#
# def add_market(x):
#     x2 = in_dex.loc[:, ["Date", "returns"]]
#     xx = pd.merge(x, x2, left_on="Date", right_on="Date", how='inner')
#     xx.columns = ["Date", "s_returns", "m_returns"]
#     return xx
# da_ta = sto_ck.groupby("Symbol")["Date", "ret"].apply(add_market).reset_index().loc[:,
#        ["Date", "Symbol", "s_returns", "m_returns"]]
# print(da_ta)
#
# def cal(x):
#     x.columns = ["x", "y"]
#     x_cov = x.rolling(60).cov().unstack()['x']['y']
#     x_var = x['y'].to_frame().rolling(60).var()
#     result = x_cov / x_var.iloc[:, 0]
#     re = pd.DataFrame(result.values, index=range(0, len(result)))
#     print(re)
#     return re
# da_ta["beta"] = da_ta.groupby("Symbol")['s_returns', 'm_returns'].apply(cal).reset_index().iloc[:, 2]
# da_ta["beta"] = da_ta.groupby("Symbol")["beta"].shift(1)
# da_ta["alpha"] = da_ta["s_returns"] - da_ta["beta"]*da_ta["m_returns"]
# da_ta.to_csv('E:\\new_beta.csv', encoding="utf-8")


# 1. Get list of stocks In_fromTop, In_fromBottom, Out_toTop, Out_toBottom at the time of underlying changing
csi_index = pd.read_csv('D:\\work\\project 3 event study\\index_comp_SH000905.csv',
                        encoding="utf-8")
csi_index_300 = pd.read_csv('D:\\work\\project 3 event study\\index_comp_SH000300.csv',
                            encoding="utf-8")
csi_index_1000 = pd.read_csv('D:\\work\\project 3 event study\\index_comp_SH000852.csv',
                        encoding="utf-8")
csi_rest = pd.read_csv('D:\\work\\project 3 event study\\index_comp_SH000985.csv',
                        encoding="utf-8")
# l1 = pd.read_csv("C:\\Users\\win\\Desktop\\预测沪深调入.csv", encoding="GBK")
# l2 = pd.read_csv("C:\\Users\\win\\Desktop\\预测沪深调出.csv", encoding="GBK")
# l3 = pd.read_csv("C:\\Users\\win\\Desktop\\预测中证调入.csv", encoding="GBK")
# l4 = pd.read_csv("C:\\Users\\win\\Desktop\\预测中证调出.csv", encoding="GBK")
alpha = pd.read_csv('E:\\new_beta.csv', encoding="GBK").iloc[:, 1:]
alpha["Date"] = alpha["Date"].apply(lambda x: str(x))
alpha["Date"] = pd.to_datetime(alpha["Date"])
csi_index["Date"] = csi_index["Date"].apply(lambda x: str(x))
csi_index_300["Date"] = csi_index_300["Date"].apply(lambda x: str(x))
csi_index_1000["Date"] = csi_index_1000["Date"].apply(lambda x: str(x))
csi_rest["Date"] = csi_rest["Date"].apply(lambda x: str(x))
da_te = csi_index["Date"]
sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
sto_ck["returns"] = sto_ck.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck["Date"] = sto_ck["Date"].apply(lambda x: str(x))
sto_ck["Date"] = pd.to_datetime(sto_ck["Date"])
xx = []
dd1 = []
dd2 = []
new_in = {}
old_out = {}
In_fromTop = {}
In_fromBottom = {}
Out_toTop = {}
Out_toBottom = {}
# new_in = pd.DataFrame()
# old_out = pd.DataFrame()
# In_fromTop = pd.DataFrame(index=np.arange(50))
# In_fromBottom = pd.DataFrame(index=np.arange(50))
# Out_toTop = pd.DataFrame(index=np.arange(50))
# Out_toBottom = pd.DataFrame(index=np.arange(50))

# IF
xx.append(csi_index_300.columns[csi_index_300.iloc[0, :] != 0])
for i in range(1, len(csi_index_300)):
    xx.append(csi_index_300.columns[csi_index_300.iloc[i, :] != 0])
    new_in[csi_index_300.iloc[i, 0]] = list(set(xx[i][1:]) - set(xx[i - 1][1:]))
    old_out[csi_index_300.iloc[i, 0]] = list(set(xx[i - 1][1:]) - set(xx[i][1:]))
new_in = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_in.items()]))
old_out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in old_out.items()]))
new_in = new_in.dropna(axis=1, thresh=5)
old_out = old_out.dropna(axis=1, thresh=5)
print(new_in)
print(old_out)


# # IC
# xx.append(csi_index.columns[csi_index.iloc[0, :] != 0])
# for i in range(1, len(csi_index)):
#     xx.append(csi_index.columns[csi_index.iloc[i, :] != 0])
#     new_in[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i][1:]) - set(xx[i - 1][1:])))
#     old_out[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i - 1][1:]) - set(xx[i][1:])))
# new_in = new_in.dropna(axis=1, how="any")
# old_out = old_out.dropna(axis=1, how="any")
#
# print(da_te)
# print(new_in)
# print(old_out)
#
# hs300_list = csi_index_300.columns
# for i in range(len(old_out.columns)):
#     test1 = csi_index_300[csi_index_300["Date"] == old_out.columns[i]][list(set(hs300_list) & set(old_out.iloc[:, i]))]
#     Out_toTop[old_out.columns[i]] = pd.Series(test1.columns[(test1 != 0).values[0]])
#     Out_toBottom[old_out.columns[i]] = pd.Series(list(set(old_out.iloc[:, i]) - set(Out_toTop[old_out.columns[i]])))
#     print(Out_toTop.iloc[:, i].count() + Out_toBottom.iloc[:, i].count())
# for i in range(len(new_in.columns)):
#     t = da_te[da_te[da_te == new_in.columns[i]].index[0] - 1]
#     test1 = csi_index_300[csi_index_300["Date"] == t][list(set(hs300_list) & set(new_in.iloc[:, i]))]
#     In_fromTop[new_in.columns[i]] = pd.Series(test1.columns[(test1 != 0).values[0]])
#     In_fromBottom[new_in.columns[i]] = pd.Series(list(set(new_in.iloc[:, i]) - set(In_fromTop[new_in.columns[i]])))
#     print(In_fromTop.iloc[:, i].count() + In_fromBottom.iloc[:, i].count())


# # CSI1000
# xx.append(csi_index_1000.columns[csi_index_1000.iloc[0, :] != 0])
# for i in range(1, len(csi_index_1000)):
#     xx.append(csi_index_1000.columns[csi_index_1000.iloc[i, :] != 0])
#     new_in[csi_index_1000.iloc[i, 0]] = list(set(xx[i][1:]) - set(xx[i - 1][1:]))
#     old_out[csi_index_1000.iloc[i, 0]] = list(set(xx[i - 1][1:]) - set(xx[i][1:]))
# new_in["20200102"].remove("SZ001914")
# old_out["20200102"].remove("SZ000043")
# new_in = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_in.items()]))
# old_out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in old_out.items()]))
# new_in = new_in.dropna(axis=1, thresh=5)
# old_out = old_out.dropna(axis=1, thresh=5)
# print(new_in)
# print(old_out)
#
# hs300_list = csi_index_300.columns
# zz500_list = csi_index.columns
# for i in range(len(old_out.columns)):
#     test1 = csi_index_300[csi_index_300["Date"] == old_out.columns[i]][list(set(hs300_list) & set(old_out.iloc[:, i]))]
#     test2 = csi_index[csi_index["Date"] == old_out.columns[i]][list(set(zz500_list) & set(old_out.iloc[:, i]))]
#     Out_toTop[old_out.columns[i]] = list(set(test1.columns[(test1 != 0).values[0]]) | set(test2.columns[(test2 != 0).values[0]]))
#     Out_toBottom[old_out.columns[i]] = list(set(old_out.iloc[:, i]) - set(Out_toTop[old_out.columns[i]]))
# for i in range(len(new_in.columns)):
#     t = da_te[da_te[da_te == new_in.columns[i]].index[0] - 1]
#     test1 = csi_index_300[csi_index_300["Date"] == t][list(set(hs300_list) & set(new_in.iloc[:, i]))]
#     test2 = csi_index[csi_index["Date"] == t][list(set(zz500_list) & set(new_in.iloc[:, i]))]
#     In_fromTop[new_in.columns[i]] = list(set(test1.columns[(test1 != 0).values[0]]) | set(test2.columns[(test2 != 0).values[0]]))
#     In_fromBottom[new_in.columns[i]] = list(set(new_in.iloc[:, i]) - set(In_fromTop[new_in.columns[i]]))
#
# Out_toTop = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Out_toTop.items()]))
# Out_toBottom = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Out_toBottom.items()]))
# In_fromTop = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in In_fromTop.items()]))
# In_fromBottom = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in In_fromBottom.items()]))

# # CSIRest
# csi_rest = csi_rest[csi_rest["Date"] >= '20141103']
# csi_index_300 = csi_index_300[csi_index_300["Date"] >= '20141103']
# csi_index1 = csi_index[csi_index["Date"] >= '20141103']
# csi_rest1 = {}
# csi_rest1["Date"] = []
# csi_rest1["Stocks"] = []
# for i in range(len(csi_rest)):
#     x1 = csi_rest.columns[csi_rest.iloc[i, :] != 0][1:]
#     x2 = csi_index_300.columns[csi_index_300.iloc[i, :] != 0][1:]
#     x3 = csi_index1.columns[csi_index1.iloc[i, :] != 0][1:]
#     x4 = csi_index_1000.columns[csi_index_1000.iloc[i, :] != 0][1:]
#     rx = list(((set(x1) - set(x2)) - set(x3)) - set(x4))
#     csi_rest1["Date"].append(csi_rest.iloc[i, 0])
#     csi_rest1["Stocks"].append(rx)
# csi_rest1 = pd.DataFrame(csi_rest1)
#
# xx.append(csi_rest1.iloc[0, 1])
# for i in range(1, len(csi_rest)):
#     xx.append(csi_rest1.iloc[i, 1])
#     new_in[csi_rest1.iloc[i, 0]] = list(set(xx[i]) - set(xx[i - 1]))
#     old_out[csi_rest1.iloc[i, 0]] = list(set(xx[i - 1]) - set(xx[i]))
# if "SZ001914" in new_in["20200102"]:
#     new_in["20200102"].remove("SZ001914")
#     old_out["20200102"].remove("SZ000043")
# if "SZ001872" in new_in["20200102"]:
#     new_in["20200102"].remove("SZ001872")
#     old_out["20200102"].remove("SZ000022")
# new_in = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_in.items()]))
# old_out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in old_out.items()]))
# new_in = new_in.dropna(axis=1, thresh=30)
# old_out = old_out.dropna(axis=1, thresh=30)
# print(new_in)
# print(old_out)
#
# hs300_list = csi_index_300.columns
# zz500_list = csi_index.columns
# zz1000_list = csi_index_1000.columns
# for i in range(len(old_out.columns)):
#     test1 = csi_index_300[csi_index_300["Date"] == old_out.columns[i]][list(set(hs300_list) & set(old_out.iloc[:, i]))]
#     test2 = csi_index1[csi_index1["Date"] == old_out.columns[i]][list(set(zz500_list) & set(old_out.iloc[:, i]))]
#     test3 = csi_index_1000[csi_index_1000["Date"] == old_out.columns[i]][list(set(zz1000_list) & set(old_out.iloc[:, i]))]
#     Out_toTop[old_out.columns[i]] = list((set(test1.columns[(test1 != 0).values[0]]) | set(test2.columns[(test2 != 0).values[0]])) | set(test3.columns[(test3 != 0).values[0]]))
#     Out_toBottom[old_out.columns[i]] = list(set(old_out.iloc[:, i]) - set(Out_toTop[old_out.columns[i]]))
# for i in range(len(new_in.columns)):
#     t = da_te[da_te[da_te == new_in.columns[i]].index[0] - 1]
#     test1 = csi_index_300[csi_index_300["Date"] == t][list(set(hs300_list) & set(new_in.iloc[:, i]))]
#     test2 = csi_index1[csi_index1["Date"] == t][list(set(zz500_list) & set(new_in.iloc[:, i]))]
#     test3 = csi_index_1000[csi_index_1000["Date"] == t][list(set(zz1000_list) & set(new_in.iloc[:, i]))]
#     In_fromTop[new_in.columns[i]] = list((set(test1.columns[(test1 != 0).values[0]]) | set(test2.columns[(test2 != 0).values[0]])) | set(test3.columns[(test3 != 0).values[0]]))
#     In_fromBottom[new_in.columns[i]] = list(set(new_in.iloc[:, i]) - set(In_fromTop[new_in.columns[i]]))
#
# Out_toTop = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Out_toTop.items()]))
# Out_toBottom = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Out_toBottom.items()]))
# In_fromTop = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in In_fromTop.items()]))
# In_fromBottom = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in In_fromBottom.items()]))

# 2. Get the exact effect day and notice day from 2010 to 2019
da_te = pd.to_datetime(csi_index["Date"])
def find_date1(year, month):
    date1 = da_te[da_te[(da_te.dt.year == year) & (da_te.dt.month == month)].index[-1] + 1]
    date2 = date1 - timedelta(days=14)
    return [date1, date2]

def find_date(year, month):
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    second_friday = [day for week in monthcal for day in week if \
                     day.weekday() == calendar.FRIDAY and \
                     day.month == month][1]
    if (da_te == second_friday).any():
        date1 = da_te[da_te[da_te == second_friday].index[0] + 1]
        date2 = date1 - timedelta(days=14)
        if(da_te == date2).any():
            date2 = date2
        elif (da_te == date2 + timedelta(days=1)).any():
            date2 = da_te[da_te == date2 + timedelta(days=1)].values[0]
        elif(da_te == date2 + timedelta(days=2)).any():
            date2 = da_te[da_te == date2 + timedelta(days=2)].values[0]
    elif (da_te == second_friday - timedelta(days=2)).any():
        date1 = da_te[da_te[da_te == second_friday - timedelta(days=2)].index[0] + 1]
        date2 = date1 - timedelta(days=14)
        if (da_te == date2).any():
            date2 = date2
        elif (da_te == date2 + timedelta(days=1)).any():
            date2 = da_te[da_te == date2 + timedelta(days=1)].values[0]
        elif (da_te == date2 + timedelta(days=2)).any():
            date2 = da_te[da_te == date2 + timedelta(days=2)].values[0]
    # For the last day 2019.12, it has not come yet
    else:
        date1 = []
        date2 = []
    return [date1, date2]

for year in range(2010, 2014):
    for month in [6, 12]:
        dd1.append(find_date1(year, month)[0])
        dd2.append(find_date1(year, month)[1])
dd1 = dd1[:-1]
dd2 = dd2[:-1]
for year in range(2013, 2020):
    for month in [6, 12]:
        dd1.append(find_date(year, month)[0])
        dd2.append(find_date(year, month)[1])
del dd1[7]
del dd2[7]
time_df = pd.DataFrame({"Effect Day": dd1, "Notice Day": dd2})
print(time_df)

# 3. calculate cumulative mean abnormal returns and plot it from pre_event_10days to post_event_20days
# Effect day + new_in
def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]

#time_df = time_df[(da_te[da_te.isin(time_df["Effect Day"]).values].index - da_te[da_te.isin(time_df["Notice Day"]).values].index) == 10]


def cal1(dataframe):
    df = pd.DataFrame()
    m = 0
    CAR = []
    T = []
    M = []

    event_date = time_df["Notice Day"].iloc[-2]
    time_period = da_te[da_te[da_te == event_date].index[0] - 10:da_te[da_te == event_date].index[0] + 21]
    stocks = dataframe.iloc[:, -2][~dataframe.iloc[:, -2].isnull()]
    for j in stocks:
        data1 = alpha[alpha["Symbol"] == j]
        tp = pd.DataFrame(time_period)
        tp = pd.DataFrame(pd.merge(tp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
        tp = tp.rename(columns={'alpha': m})
        df = pd.concat([df, tp], axis=1)
        m = m + 1
    print(df)
    # for i in range(df.shape[0]):
    #     T.append(np.nanmean(df.iloc[i, :]) * np.sqrt(df.shape[1]) / np.nanstd(df.iloc[i, :]))
    CAR = np.cumsum(df.mean(axis=1)) * 10000
    T = df.mean(axis=1) * 10000
    M = df.median(axis=1) * 10000
    # d = mf[(mf["date"] >= 20180528) & (mf["date"] <= 20180611) & (mf["StockID"].isin(stocks))]
    # nd = d.groupby("date")["buy_lg_vol", "buy_elg_vol", "sell_lg_vol", "sell_elg_vol"].sum()
    # nd["sum_buy"] = nd["buy_lg_vol"] + nd["buy_elg_vol"]
    # nd["sum_sell"] = nd["sell_lg_vol"] + nd["sell_elg_vol"]
    # nd_time = pd.Series(nd.index.values).apply(lambda x: str(x))
    # nd_time = pd.to_datetime(nd_time)
    # nd = nd.set_index(nd_time)
    #fig, ax1 = plt.subplots()
    #plt.plot(nd["sum_buy"], 'b-', label="total buy volume")
    #plt.plot(nd["sum_sell"], 'r-', label="total sell volume")
    #plt.xticks(rotation=90)
    #plt.legend()
    #plt.show()

    print(df.mean(axis=1)[:10].sum()*10000)
    print(df.mean(axis=1)[11:20].sum()*10000)
    print(df.mean(axis=1)[21:].sum()*10000)

    print(df.mean(axis=1)[:5].sum() * 10000)
    print(df.mean(axis=1)[5:10].sum() * 10000)
    print(df.mean(axis=1)[11:15].sum() * 10000)
    print(df.mean(axis=1)[15:20].sum() * 10000)
    print(df.mean(axis=1)[21:26].sum() * 10000)
    print(df.mean(axis=1)[26:].sum() * 10000)



    return pd.DataFrame({"CAR": CAR, "T value": T, "stock number": np.repeat(stocks.size, CAR.size), "Median_alpha": M})



def cal2(day, dataframe):
    df = pd.DataFrame()
    m = 0
    CAR = []
    T = []
    for i in range(18):
        event_date = time_df[day].iloc[i]
        print(event_date)
        time_period = da_te[da_te[da_te == event_date].index[0] - 10:da_te[da_te == event_date].index[0] + 21]
        stocks = dataframe.iloc[:, i+1][~dataframe.iloc[:, i+1].isnull()]
        for j in stocks:
            data1 = alpha[alpha["Symbol"] == j]
            tp = pd.DataFrame(time_period)
            tp = pd.DataFrame(pd.merge(tp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            tp = tp.rename(columns={'alpha': m})
            df = pd.concat([df, tp], axis=1)
            m = m + 1
    print(df)
    print(df.shape[1])
    T = df.mean(axis=1) * 10000
    CAR = np.cumsum(df.mean(axis=1)) * 10000

    print(df.mean(axis=1)[:10].sum()*10000)
    #print(df.mean(axis=1)[:10].mean() * 10000)
    #print((df.mean(axis=1)[9] - df.mean(axis=1)[0]) * 10000)
    #print(df.mean(axis=1)[10]*10000)
    print(df.mean(axis=1)[11:19].sum()*10000)
    #print(df.mean(axis=1)[11:19].mean() * 10000)
    #print((df.mean(axis=1)[18] - df.mean(axis=1)[11]) * 10000)
    #print((df.mean(axis=1)[19]) * 10000)
    print(df.mean(axis=1)[20:].sum() * 10000)
    #print(df.mean(axis=1)[20:].mean() * 10000)
    #print((df.mean(axis=1)[30] - df.mean(axis=1)[20]) * 10000)

    # print(df.mean(axis=1)[0] * 10000)
    # #print(df.mean(axis=1)[1] * 10000)
    # print(df.mean(axis=1)[2:10].sum() * 10000)
    # #print(df.mean(axis=1)[2:10].mean() * 10000)
    # #print((df.mean(axis=1)[9] - df.mean(axis=1)[2]) * 10000)
    # #print((df.mean(axis=1)[10]) * 10000)
    # print(df.mean(axis=1)[11:].sum() * 10000)
    # #print(df.mean(axis=1)[11:].mean() * 10000)
    # #print((df.mean(axis=1)[30] - df.mean(axis=1)[11]) * 10000)

    print(df.mean(axis=1)[:5].sum() * 10000)
    print(df.mean(axis=1)[5:10].sum() * 10000)
    # print(df.mean(axis=1)[:10].mean() * 10000)
    # print((df.mean(axis=1)[9] - df.mean(axis=1)[0]) * 10000)
    # print(df.mean(axis=1)[10]*10000)
    print(df.mean(axis=1)[11:15].sum() * 10000)
    print(df.mean(axis=1)[15:19].sum() * 10000)
    # print(df.mean(axis=1)[11:19].mean() * 10000)
    # print((df.mean(axis=1)[18] - df.mean(axis=1)[11]) * 10000)
    # print((df.mean(axis=1)[19]) * 10000)
    print(df.mean(axis=1)[20:25].sum() * 10000)
    print(df.mean(axis=1)[25:].sum() * 10000)
    # print(df.mean(axis=1)[20:].mean() * 10000)
    # print((df.mean(axis=1)[30] - df.mean(axis=1)[20]) * 10000)



    return pd.DataFrame({"CAR": CAR, "T value": T, "stock number": np.repeat(stocks.size, CAR.size)})


x = range(-10, 21)
sns.set()
fig, ax1 = plt.subplots(2, 2, figsize=(8, 8))
# fig, ax1 = plt.subplots(1, 2, figsize=(8, 8))

# r1 = cal1(In_fromBottom)
r1 = cal2("Notice Day", new_in)
# r1 = cal1(new_in)
len = r1.loc[0, "stock number"]
ax1[0, 0].axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
ax1[0, 0].axvspan(0, 10, facecolor='green', alpha=0.5)
ax1[0, 0].axvspan(10, 20, facecolor='darkgreen', alpha=0.5)
ax1[0, 0].set_ylabel('Cumulative Alpha', fontname="Arial", fontsize=8)
ax1[0, 0].plot(x, r1["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
ax1[0, 0].tick_params('y')
ax1[0, 0].tick_params(labelsize=8)
ax1[0, 0].axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax2 = ax1[0, 0].twinx()
ax2.set_ylabel('Mean Alpha', fontname="Arial", fontsize=8)
ax2.bar(x, r1["T value"], color='salmon', alpha=0.8)
# ax2.scatter(x, r1["Median_alpha"], color='red', alpha=0.8, s=4)
# if len < 30:
#     ax2.axhline(y=stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
#     ax2.axhline(y=-stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6,
#                 linewidth=1)
# else:
#     ax2.axhline(y=1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
#     ax2.axhline(y=-1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6,
#                 linewidth=1)
ax2.tick_params('y')
ax2.tick_params(labelsize=8)
# plt.title("In_fromBottom_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
plt.title("Cumulative Average alpha for stocks In_fromBottom before/after notice day", fontname="Arial", fontsize=10)
# plt.title("New_in_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
ax1[0, 0].grid(True)
ax2.grid(None)
ax1[0, 0].set_ylim(-1000, 200)
ax2.set_ylim(-200, 200)
align_yaxis_np(ax1[0, 0], ax2)



# r2 = cal1(In_fromTop)
r2 = cal2("Notice Day", old_out)
# r2 = cal1(old_out)
len = r2.loc[0, "stock number"]
ax1[0, 1].axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
ax1[0, 1].axvspan(0, 10, facecolor='green', alpha=0.5)
ax1[0, 1].axvspan(10, 20, facecolor='darkgreen', alpha=0.5)
ax1[0, 1].set_ylabel('Cumulative Alpha', fontname="Arial", fontsize=8)
ax1[0, 1].plot(x, r2["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
ax1[0, 1].tick_params('y')
ax1[0, 1].tick_params(labelsize=8)
ax1[0, 1].axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax2 = ax1[0, 1].twinx()
ax2.set_ylabel('Mean Alpha', fontname="Arial", fontsize=8)
ax2.bar(x, r2["T value"], color='salmon', alpha=0.8)
# ax2.scatter(x, r2["Median_alpha"], color='red', alpha=0.8, s=4)
# if len < 30:
#     ax2.axhline(y=stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
#     ax2.axhline(y=-stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6,
#                 linewidth=1)
# else:
#     ax2.axhline(y=1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
#     ax2.axhline(y=-1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6,
#                 linewidth=1)
ax2.tick_params('y')
ax2.tick_params(labelsize=8)
# plt.title("In_fromTop_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
plt.title("Cumulative Average alpha for stocks In_fromTop before/after notice day", fontname="Arial", fontsize=10)
# plt.title("Old_out_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
ax1[0, 1].grid(True)
ax2.grid(None)
ax1[0, 1].set_ylim(-1000, 200)
ax2.set_ylim(-200, 200)
align_yaxis_np(ax1[0, 1], ax2)


# r3 = cal1(Out_toTop)
# # r3 = cal2("Notice Day", Out_toTop)
# len = r3.loc[0, "stock number"]
# ax1[1, 0].axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
# ax1[1, 0].axvspan(0, 9, facecolor='green', alpha=0.5)
# ax1[1, 0].axvspan(9, 20, facecolor='darkgreen', alpha=0.5)
# ax1[1, 0].set_ylabel('Cumulative Alpha', fontname="Arial", fontsize=8)
# ax1[1, 0].plot(x, r3["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
# ax1[1, 0].tick_params('y')
# ax1[1, 0].tick_params(labelsize=8)
# ax1[1, 0].axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# ax2 = ax1[1, 0].twinx()
# ax2.set_ylabel('Mean Alpha', fontname="Arial", fontsize=8)
# ax2.bar(x, r3["T value"], color='salmon', alpha=0.8)
# # ax2.scatter(x, r3["Median_alpha"], color='red', alpha=0.8, s=4)
# # if len < 30:
# #     ax2.axhline(y=stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# #     ax2.axhline(y=-stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6,
# #                 linewidth=1)
# # else:
# #     ax2.axhline(y=1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# #     ax2.axhline(y=-1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6,
# #                 linewidth=1)
# ax2.tick_params('y')
# ax2.tick_params(labelsize=8)
# # plt.title("Out_toTop_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
# plt.title("Cumulative Average alpha for stocks Out_toTop before/after notice day", fontname="Arial", fontsize=10)
# ax1[1, 0].grid(True)
# ax2.grid(None)
# ax1[1, 0].set_ylim(-1000, 200)
# ax2.set_ylim(-200, 200)
# align_yaxis_np(ax1[1, 0], ax2)
#
#
# r4 = cal1(Out_toBottom)
# # r4 = cal2("Notice Day", Out_toBottom)
# len = r4.loc[0, "stock number"]
# ax1[1, 1].axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
# ax1[1, 1].axvspan(0, 9, facecolor='green', alpha=0.5)
# ax1[1, 1].axvspan(9, 20, facecolor='darkgreen', alpha=0.5)
# ax1[1, 1].set_ylabel('Cumulative Alpha', fontname="Arial", fontsize=8)
# ax1[1, 1].plot(x, r4["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
# ax1[1, 1].tick_params('y')
# ax1[1, 1].tick_params(labelsize=8)
# ax1[1, 1].axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# ax2 = ax1[1, 1].twinx()
# ax2.set_ylabel('Mean Alpha', fontname="Arial", fontsize=8)
# ax2.bar(x, r4["T value"], color='salmon', alpha=0.8)
# # ax2.scatter(x, r4["Median_alpha"], color='red', alpha=0.8, s=4)
# # if len < 30:
# #     ax2.axhline(y=stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# #     ax2.axhline(y=-stats.t.ppf(1 - 0.025, len - 1), color='green', linestyle=(0, (1, 1)), alpha=0.6,
# #                 linewidth=1)
# # else:
# #     ax2.axhline(y=1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# #     ax2.axhline(y=-1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6,
# #                 linewidth=1)
# ax2.tick_params('y')
# ax2.tick_params(labelsize=8)
# # plt.title("Out_toBottom_201906_" + str(len) + "stocks", fontname="Arial", fontsize=10)
# plt.title("Cumulative Average alpha for stocks Out_toBottom before/after notice day", fontname="Arial", fontsize=10)
# ax1[1, 1].grid(True)
# ax2.grid(None)
# ax1[1, 1].set_ylim(-1000, 200)
# ax2.set_ylim(-200, 200)
# align_yaxis_np(ax1[1, 1], ax2)
#
# fig.tight_layout()
#
# plt.show()
#
# # time_period = da_te[da_te[da_te == time_df["Notice Day"].iloc[-1]].index[0] - 10:
# #                     da_te[da_te == time_df["Notice Day"].iloc[-1]].index[0] + 21]
# # IC["ret"] = IC["close"]/IC["close"].shift(1)-1
# # CSI["ret"] = CSI["close"]/CSI["close"].shift(1)-1
# # IC["Date"] = IC["Date"].apply(lambda x: str(x))
# # IC["Date"] = pd.to_datetime(IC["Date"])
# # CSI["Date"] = CSI["Date"].apply(lambda x: str(x))
# # CSI["Date"] = pd.to_datetime(CSI["Date"])
# # diff = (CSI.loc[CSI["Date"].isin(time_period), "ret"].values - IC.loc[IC["Date"].isin(time_period), "ret"].values)*10000
# # x = range(-10, 21)
# # sns.set()
# # fig, ax = plt.subplots()
# # ax.axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
# # ax.axvspan(0, 10, facecolor='green', alpha=0.5)
# # ax.axvspan(10, 20, facecolor='darkgreen', alpha=0.5)
# # ax.set_ylabel('spreads', fontname="Arial", fontsize=8)
# # ax.bar(x, diff, color='salmon', alpha=0.8, label="spreads")
# # ax.plot(x, diff.cumsum(), marker='.', color='blue', alpha=1, linewidth=1, markersize=2, label="cumulative spreads")
# # ax.legend(loc='upper right')
# # ax.set_title("Spreads between CSI and IC on 201912")
# # ax.grid(True)
# # plt.show()