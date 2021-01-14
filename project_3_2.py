import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import timedelta
import calendar
import seaborn as sns
import datetime

# 1. Get list of stocks new_in, old_out, old_in at the time of underlying changing
csi_index = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv', encoding="utf-8")
csi_index_300 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000300.csv', encoding="utf-8")
csi_index_1000 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000852.csv', encoding="utf-8")
csi_index_all = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000985.csv', encoding="utf-8")
ST_stocks = pd.read_csv('C:\\Users\\win\\Downloads\\STlist.csv', encoding="utf-8")
t_s = pd.read_csv('E:\\trade_status.csv', encoding="utf-8").iloc[:, 1:]
beta1 = pd.read_csv('E:\\beta1.csv', encoding="utf-8").iloc[:, 1:]
beta2 = pd.read_csv('E:\\beta2.csv', encoding="utf-8").iloc[:, 1:]
beta1["Date"] = beta1["Date"].apply(lambda x: str(x))
beta1["Date"] = pd.to_datetime(beta1["Date"])
beta2["Date"] = beta2["Date"].apply(lambda x: str(x))
beta2["Date"] = pd.to_datetime(beta2["Date"])
csi_index["Date"] = csi_index["Date"].apply(lambda x: str(x))
csi_index_300["Date"] = csi_index_300["Date"].apply(lambda x: str(x))
csi_index_1000["Date"] = csi_index_1000["Date"].apply(lambda x: str(x))
csi_index_all["Date"] = csi_index_all["Date"].apply(lambda x: str(x))
da_te = pd.to_datetime(csi_index["Date"])
sto_ck = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]
sto_ck["returns"] = sto_ck.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck["Date"] = sto_ck["Date"].apply(lambda x: str(x))
sto_ck["Date"] = pd.to_datetime(sto_ck["Date"])
xx = []
dd1 = []
dd2 = []
xx.append(csi_index.columns[csi_index.iloc[0, :] != 0])
new_in = pd.DataFrame()
old_out = pd.DataFrame()
old_in = pd.DataFrame()
old_out_hs300 = pd.DataFrame(index=np.arange(50))
old_out_zz1000 = pd.DataFrame(index=np.arange(50))
old_out_zzall = pd.DataFrame(index=np.arange(50))
old_out_else = pd.DataFrame(index=np.arange(50))
for i in range(1, len(csi_index)):
    xx.append(csi_index.columns[csi_index.iloc[i, :] != 0])
    new_in[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i][1:]) - set(xx[i - 1][1:])))
    old_out[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i - 1][1:]) - set(xx[i][1:])))
    old_in[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i - 1][1:]) & set(xx[i][1:])))
new_in = new_in.dropna(axis=1, how="any")
old_out = old_out.dropna(axis=1, how="any")
old_in = old_in[new_in.columns]
old_in = old_in.dropna(axis=0, how="all")



#xx1 = []
#xx1.append(csi_index_300.columns[csi_index_300.iloc[0, :] != 0])
#new_in1 = pd.DataFrame()
#old_out1 = pd.DataFrame()
#old_in1 = pd.DataFrame()
#for i in range(1, len(csi_index_300)):
    #xx1.append(csi_index_300.columns[csi_index_300.iloc[i, :] != 0])
    #df1 = pd.DataFrame({csi_index_300.iloc[i, 0]: list(set(xx1[i][1:]) - set(xx1[i - 1][1:]))})
    #new_in1 = pd.concat([new_in1, df1], ignore_index=True, axis=1)
    #df2 = pd.DataFrame({csi_index_300.iloc[i, 0]: list(set(xx1[i - 1][1:]) - set(xx1[i][1:]))})
    #old_out1 = pd.concat([old_out1, df2], ignore_index=True, axis=1)
    #df3 = pd.DataFrame({csi_index_300.iloc[i, 0]: list(set(xx1[i - 1][1:]) & set(xx1[i][1:]))})
    #old_in1 = pd.concat([old_in1, df3], ignore_index=True, axis=1)
#new_in1.columns = csi_index_300.iloc[1:, 0]
#old_out1.columns = csi_index_300.iloc[1:, 0]
#old_in1.columns = csi_index_300.iloc[1:, 0]
#new_in1 = new_in1[new_in.columns]
#new_in1 = new_in1.dropna(axis=0, how="all")
#old_out1 = old_out1[new_in.columns]
#old_out1 = old_out1.dropna(axis=0, how="all")
#old_in1 = old_in1[new_in.columns]
#old_in1 = old_in1.dropna(axis=0, how="all")



print(da_te)
print(new_in)
print(old_out)
print(old_in)

hs300_list = csi_index_300.columns
zz1000_list = csi_index_1000.columns
zzall_list = csi_index_all.columns
for i in range(len(old_out.columns)):
    test1 = csi_index_300[csi_index_300["Date"] == old_out.columns[i]][list(set(hs300_list) & set(old_out.iloc[:,i]))]
    old_out_hs300[old_out.columns[i]] = pd.Series(test1.columns[(test1!=0).values[0]])
    test2 = csi_index_1000[csi_index_1000["Date"] == old_out.columns[i]][list(set(zz1000_list) & set(old_out.iloc[:,i]) - set(old_out_hs300.iloc[:,i]))]
    old_out_zz1000[old_out.columns[i]] = pd.Series(test2.columns[(test2!=0).values[0]])
    test3 = csi_index_all[csi_index_all["Date"] == old_out.columns[i]][list(set(zzall_list) & set(old_out.iloc[:,i]))]
    old_out_zzall[old_out.columns[i]] = pd.Series(test3.columns[(test3!=0).values[0]])
    old_out_else[old_out.columns[i]] = pd.Series(list(set(old_out.iloc[:,i]) - set(old_out_hs300.iloc[:,i]) - set(old_out_zz1000.iloc[:,i]) - set(old_out_zzall.iloc[:,i])))
    print(old_out_else.iloc[:,i].count() + old_out_hs300.iloc[:,i].count() + old_out_zz1000.iloc[:,i].count() + old_out_zzall.iloc[:,i].count())


# 2. Get the exact effect day and notice day from 2010 to 2019
def find_date1(year, month):
    date1 = da_te[da_te[(da_te.dt.year == year) & (da_te.dt.month == month)].index[-1] + 1]
    date2 = date1 - timedelta(days=14)
    date2 = da_te[da_te[da_te==date2].index[0]+1]
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
            date2 = da_te[da_te[da_te == date2].index[0] + 1]
        elif (da_te == date2 + timedelta(days=1)).any():
            date2 = da_te[da_te[da_te == date2 + timedelta(days=1)].index[0] + 1]
        elif(da_te == date2 + timedelta(days=2)).any():
            date2 = da_te[da_te[da_te == date2 + timedelta(days=2)].index[0] + 1]
    elif (da_te == second_friday - timedelta(days=2)).any():
        date1 = da_te[da_te[da_te == second_friday - timedelta(days=2)].index[0] + 1]
        date2 = date1 - timedelta(days=14)
        if (da_te == date2).any():
            date2 = da_te[da_te[da_te == date2].index[0] + 1]
        elif (da_te == date2 + timedelta(days=1)).any():
            date2 = da_te[da_te[da_te == date2 + timedelta(days=1)].index[0] + 1]
        elif (da_te == date2 + timedelta(days=2)).any():
            date2 = da_te[da_te[da_te == date2 + timedelta(days=2)].index[0] + 1]
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
del dd1[19]
del dd2[7]
del dd2[19]
time_df = pd.DataFrame({"Effect Day": dd1, "Notice Day": dd2})
print(time_df)

# 3. calculate cumulative mean abnormal returns and plot it from pre_event_20days to post_event_20days
# Effect day + new_in
def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


def cal(day, dataframe):
    df = pd.DataFrame()
    m = 0
    CAR = []
    T = []
    for i in range(len(time_df)):
        event_date = time_df[day].iloc[i]
        time_period = da_te[da_te[da_te == event_date].index[0] - 20:da_te[da_te == event_date].index[0] + 21]
        stocks = dataframe.iloc[:, i + 1][~dataframe.iloc[:,i + 1].isnull()]
        for j in stocks:
            data1 = beta2[beta2["Symbol"] == j]
            tp = pd.DataFrame(time_period)
            tp = pd.DataFrame(pd.merge(tp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            tp = tp.rename(columns={'alpha': m})
            df = pd.concat([df, tp], axis=1)
            m = m + 1
    print(df)
    n = int((df.shape[0] - 1) / 2)
    for i in range(df.shape[0]):
        s_um = 0
        for j in range(df.shape[1]):
            if i < n:
                car = np.nansum(df.iloc[i:n, j])
            else:
                car = np.nansum(df.iloc[n:i + 1, j])
            s_um = s_um + car
        CAR.append(s_um / df.shape[1])
        T.append(np.nanmean(df.iloc[i, :]) * np.sqrt(df.shape[1]) / np.nanstd(df.iloc[i, :]))
    v1 = np.insert(CAR, 20, 0)
    v2 = np.insert(T, 20, 0)
    x = range(-20, 22)
    sns.set()
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('t value', fontname="Arial", fontsize=6)
    ax1.bar(x, v2, color='salmon', alpha=0.8)
    ax1.tick_params('y')
    ax1.tick_params(labelsize=6)
    ax1.axhline(y=1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
    ax1.axhline(y=-1.96, color='green', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
    ax1.axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
    ax1.axvline(x=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Abnormal Returns', fontname="Arial", fontsize=6)
    ax2.plot(x, v1, marker='.', color='blue', alpha=0.5, linewidth=1, markersize=2)
    ax2.tick_params('y')
    ax2.tick_params(labelsize=6)

    plt.title("Cumulative Average alpha before/after Implementation for old_out stocks", fontname="Arial", fontsize=8)
    ax1.grid(True)
    ax2.grid(None)
    align_yaxis_np(ax1, ax2)
    fig.tight_layout()
    plt.show()

#cal("Effect Day", old_out_hs300)
#cal("Notice Day", old_out_hs300)
#cal("Effect Day", old_out_zz1000)
#cal("Effect Day", old_out_zzall)
#cal("Effect Day", old_out_else)
#print((len(old_out_hs300)-old_out_hs300.isnull().sum()).describe())
#print((len(old_out_zz1000)-old_out_zz1000.isnull().sum()).describe())
#print((len(old_out_zzall)-old_out_zzall.isnull().sum()).describe())
#print((len(old_out_else)-old_out_else.isnull().sum()).describe())

# CSI500 old list
total_list = pd.concat([new_in.iloc[:, -1], old_in.iloc[:, -1]])
# CSI500 new list
total_list1 = pd.concat([new_in.iloc[:, -1], old_in.iloc[:, -1]])
total_list2 = old_in.iloc[:, -1]
total_list3 = new_in.iloc[:, -1]
# 第一部分：筛选沪深300样本指数成分股
# 1. 筛选样本空间

all_data = sto_ck[(sto_ck['Date'] >= "20181101") & (sto_ck['Date'] <= "20191031")].loc[:, ["Date", "Symbol",
        "Name", "TotalValue", "MarketValue", "Totalshares", "Marketshares", "close", "amt", "ListDays"]]
all_data.loc[all_data["Symbol"] == "SH600600", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600600", "MarketValue"]
all_data.loc[all_data["Symbol"] == "SH600600", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600600", "Marketshares"]

all_data.loc[all_data["Symbol"] == "SZ000039", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SZ000039", "MarketValue"]
all_data.loc[all_data["Symbol"] == "SZ000039", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SZ000039", "Marketshares"]

all_data.loc[all_data["Symbol"] == "SH601598", "Totalshares"] = 5255916875
all_data.loc[all_data["Symbol"] == "SH601598", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH601598", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH601598", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600801", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600801", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600801", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600801", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH600612", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600612", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600612", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600612", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH600548", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600548", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600548", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600548", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH600685", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600685", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600685", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600685", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH601375", "Totalshares"] = 2673705700
all_data.loc[all_data["Symbol"] == "SH601375", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH601375", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH601375", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600320", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600320", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600320", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600320", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH601866", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH601866", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH601866", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH601866", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SZ002936", "Totalshares"] = 4403931900
all_data.loc[all_data["Symbol"] == "SZ002936", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SZ002936", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SZ002936", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600875", "Totalshares"] = 2750803431
all_data.loc[all_data["Symbol"] == "SH600875", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH600875", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH600875", "close"])/10000

all_data.loc[all_data["Symbol"] == "SZ002948", "Totalshares"] = 2746655020
all_data.loc[all_data["Symbol"] == "SZ002948", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SZ002948", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SZ002948", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH601869", "Totalshares"] = 406338314
all_data.loc[all_data["Symbol"] == "SH601869", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH601869", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH601869", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600845", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600845", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600845", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600845", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SZ000429", "Totalshares"] = 1742056126
all_data.loc[all_data["Symbol"] == "SZ000429", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SZ000429", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SZ000429", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600635", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600635", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600635", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600635", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SH600604", "Totalshares"] = 1407454804
all_data.loc[all_data["Symbol"] == "SH600604", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH600604", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH600604", "close"])/10000

all_data.loc[all_data["Symbol"] == "SZ001872", "Totalshares"] = 1613516948
all_data.loc[all_data["Symbol"] == "SZ001872", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SZ001872", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SZ001872", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH601598", "Totalshares"] = 5255916875
all_data.loc[all_data["Symbol"] == "SH601598", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SH601598", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SH601598", "close"])/10000

all_data.loc[all_data["Symbol"] == "SH600611", "Totalshares"] = \
        all_data.loc[all_data["Symbol"] == "SH600611", "Marketshares"]
all_data.loc[all_data["Symbol"] == "SH600611", "TotalValue"] = \
        all_data.loc[all_data["Symbol"] == "SH600611", "MarketValue"]

all_data.loc[all_data["Symbol"] == "SZ000541", "Totalshares"] = 1077274404
all_data.loc[all_data["Symbol"] == "SZ000541", "TotalValue"] = (all_data.loc[
        all_data["Symbol"] == "SZ000541", "Totalshares"] * all_data.loc[all_data["Symbol"] == "SZ000541", "close"])/10000


t_s = t_s[(t_s["Date"] >= 20181101) & (t_s["Date"] <= 20191031)]
t_s.replace(1, np.nan, inplace=True)
t_s.replace(0, 1, inplace=True)
t_s.replace(np.nan, 0, inplace=True)
la_st = all_data.groupby("Symbol").last().reset_index()
old_list = hs300_list[(csi_index_300[csi_index_300["Date"] == old_in.columns[-1]] != 0).values[0]][1:]
la_st["CSI500"] = np.nan
for i in range(len(total_list)):
    la_st.loc[la_st["Symbol"] == total_list.iloc[i], ["CSI500"]] = 1
la_st["HS300"] = np.nan
for i in range(len(old_list)):
    la_st.loc[la_st["Symbol"] == old_list[i], ["HS300"]] = 1
# 非创业板股票上市时间超过一个季度，创业板股票上市时间超过三年
l1 = all_data.groupby("Symbol")["TotalValue"].mean().reset_index()
l1 = l1.rename(columns={'TotalValue': 'daily_tv'})
l2 = all_data.groupby("Symbol")["amt"].mean().reset_index()
l2 = l2.rename(columns={'amt': 'daily_amt'})
l3 = all_data.groupby("Symbol").size().reset_index(name='TradeDays')
la_st = pd.merge(la_st, l1, left_on="Symbol", right_on="Symbol")
la_st = pd.merge(la_st, l2, left_on="Symbol", right_on="Symbol")
la_st = pd.merge(la_st, l3, left_on="Symbol", right_on="Symbol")
la_st1 = la_st
# la_st = la_st[la_st["Date"] == "20191031"]
#la_st = la_st[((la_st["ListDays"] < 243) & (la_st["ListDays"] - la_st["TradeDays"] < 25)) |
#              ((la_st["ListDays"] >= 243) & (la_st["TradeDays"] >= 213))]
cyb = la_st[(la_st["Symbol"].str[:3] == "SZ3") & (la_st["ListDays"] > 750)]["Symbol"]
fcyb = la_st[(la_st["Symbol"].str[:3] != "SZ3") & (la_st["ListDays"] > 60) &
             (la_st["Name"].str.contains("ST") == False) & (la_st["Symbol"].str[:5] != "SH688") &
             (la_st["Name"].str.contains("退") == False)]["Symbol"]  # 2691只股票，不存在上市小于一季度日均总市值前30的情况
la_st = la_st[la_st["Symbol"].isin(cyb) | la_st["Symbol"].isin(fcyb)]
t_s = t_s[t_s["Symbol"].isin(la_st["Symbol"])]
li_st = t_s.groupby("Symbol")["trade_stats"].rolling(25).sum().reset_index().groupby("Symbol")["trade_stats"].max()
de_lete = li_st[li_st >= 25].index.values
la_st = la_st[~la_st["Symbol"].isin(["SZ000418", "SZ300104", "SZ300216", "SH600270"])]


la_st["tv_rank"] = la_st["daily_tv"].rank(ascending=False)
la_st["amt_rank"] = la_st["daily_amt"].rank(ascending=False)
# 2. 按照日均成交金额筛选
data1 = all_data[all_data["Symbol"].isin(la_st["Symbol"])]
data1 = data1[data1["ListDays"] >= 4]
l1 = data1.groupby("Symbol")["TotalValue"].mean().reset_index()
l1 = l1.rename(columns={'TotalValue': 'daily_tv'})
l2 = data1.groupby("Symbol")["amt"].mean().reset_index()
l2 = l2.rename(columns={'amt': 'daily_amt'})
data1 = data1.groupby("Symbol").last().reset_index()
data1 = pd.merge(data1, l1, left_on="Symbol", right_on="Symbol")
data1 = pd.merge(data1, l2, left_on="Symbol", right_on="Symbol")
data1["amt_rank"] = data1["daily_amt"].rank(ascending=False)
la_st = pd.merge(la_st, data1.loc[:, ["Symbol", "daily_amt", "amt_rank", "daily_tv"]],
                 left_on="Symbol", right_on="Symbol", how="left")
la_st = la_st.sort_values(by=["tv_rank"])
hs = pd.concat([la_st[(la_st["HS300"] == 1) & (la_st["amt_rank_y"] <= len(la_st)*0.6)], la_st[(np.isnan(la_st["HS300"]))
                                            & (la_st["amt_rank_y"] <= len(la_st)*0.5)]])


# 3. 按照日均总市值筛选
hs["tv_rank"] = hs["daily_tv_y"].rank(ascending=False)
hs = hs.sort_values(by=['tv_rank'])
n_i = hs[(np.isnan(hs["HS300"])) & (hs["tv_rank"] <= 240)]
n_i = n_i[~n_i["Symbol"].isin(["SZ002157"])]
len1 = len(n_i)
o_i = hs[(hs["HS300"] == 1) & (hs["tv_rank"] <= 360)]
len2 = len(o_i)
if len1 <= 30:
    hs1 = n_i.iloc[:len1, :]
    if len2 >= 300 - len1:
        hs2 = o_i.iloc[:(300 - len1), :]
    elif (len2 >= 270) & (len2 < 300 - len1):
        hs1 = n_i.iloc[:(300 - len2), :]
        hs2 = o_i.iloc[:len2, :]
    else:
        hs1 = n_i.iloc[:30, :]
        hs2 = o_i.iloc[:270, :]
else:
    hs1 = n_i.iloc[:30, :]
    hs2 = hs[hs["HS300"] == 1].iloc[:270, :]

#n1 = sum(data1["CSI500"] == 1)
list1 = hs300_list[(csi_index_300[csi_index_300["Date"] == old_in.columns[-1]] != 0).values[0]][1:]
list2 = pd.concat([hs1, hs2])["Symbol"]
print('第一轮预测后沪深300指数与实际值相同的股票数为: %d' % (len(set(list1) & set(list2))))
dff3 = la_st1[la_st1["Symbol"].isin(list(set(list1) - set(old_list)))].loc[:,
       ["Symbol", "Name", "daily_tv", "daily_amt"]]
#dff3.to_csv("E:\\实际沪深调入.csv", encoding = "GBK")
dff4 = la_st1[la_st1["Symbol"].isin(list(set(old_list) - set(list1)))].loc[:,
       ["Symbol", "Name", "daily_tv", "daily_amt"]]
#dff4.to_csv("E:\\实际沪深调出.csv", encoding = "GBK")

# test!!!!!!!
#hs1 = hs1.drop(hs1[hs1["Symbol"] == "SH600600"].index | hs1[hs1["Symbol"] == "SZ000039"].index |
#               hs1[hs1["Symbol"] == "SH601598"].index | hs1[hs1["Symbol"] == "SH600801"].index |
#               hs1[hs1["Symbol"] == "SZ002157"].index)
#hs2 = hs[hs["HS300"] == 1].iloc[:284, :]
dff1 = hs1.loc[:, ["Symbol", "Name", "daily_tv", "daily_amt"]]
dff1.to_csv("E:\\预测沪深调入.csv", encoding = "GBK")
dff2 = la_st1[la_st1["Symbol"].isin(list(set(old_list) - set(list2)))].loc[:,
       ["Symbol", "Name", "daily_tv", "daily_amt"]]
dff2.to_csv("E:\\预测沪深调出.csv", encoding = "GBK")
li_st = list(set(hs2["Symbol"]) | set(la_st[la_st["tv_rank"] <= 300]["Symbol"]) | set(hs1["Symbol"]))
data2 = la_st.drop(la_st[la_st["Symbol"].isin(li_st)].index, axis=0)
#data2 = data1.drop(hs2.index, axis=0)
#print('剔除沪深300样本后显示，原股票升级数: %d' % (n1 - sum(data2["CSI500"] == 1)))


# 第二部分：筛选中证500样本指数成分股
data2["tv_rank"] = data2["daily_tv_x"].rank(ascending=False)
data2["amt_rank"] = data2["daily_amt_x"].rank(ascending=False)
# 1. 按照日均成交金额筛选
data2 = pd.concat([data2[(data2["CSI500"] == 1) & (data2["amt_rank"] < len(data2)*0.9)],
                   data2[(np.isnan(data2["CSI500"])) & (data2["amt_rank"] < len(data2)*0.8)]])
# 2. 按照日均总市值筛选
data2["tv_rank"] = data2["daily_tv_x"].rank(ascending=False)
data2 = data2.sort_values(by=['tv_rank'])
n_i = data2[(np.isnan(data2["CSI500"])) & (data2["tv_rank"] <= 400)]
n_i = n_i[~n_i["Symbol"].isin(["SZ002310"])]
n_i = n_i[~n_i["Symbol"].isin(["SZ000503"])]
n_i = n_i[~n_i["Symbol"].isin(["SZ000550"])]
n_i = n_i[~n_i["Symbol"].isin(["SZ002124"])]
n_i = n_i[~n_i["Symbol"].isin(["SZ000839"])]
o_i = data2[(data2["CSI500"] == 1) & (data2["tv_rank"] <= 600)]
len3 = len(n_i)
len4 = len(o_i)

if len3 <= 50:
    db1 = n_i.iloc[:len3, :]
    if len4 >= 500 - len3:
        db2 = o_i.iloc[:(500 - len3), :]
    elif (len4 >= 450) & (len4 < 500 - len3):
        db1 = n_i.iloc[:(500 - len4), :]
        db2 = o_i.iloc[:len4, :]
    else:
        db1 = n_i.iloc[:50, :]
        db2 = o_i.iloc[:450, :]
else:
    db1 = n_i.iloc[:50, :]
    db2 = data2[data2["CSI500"] == 1].iloc[:450, :]


new_list = pd.concat([db1, db2])["Symbol"]
dff1 = db1.loc[:, ["Symbol", "Name", "daily_tv_x", "daily_amt_x"]]
dff1.to_csv("E:\\预测中证调入.csv", encoding = "GBK")
all = all_data.groupby("Symbol").last().reset_index()
dff2 = la_st1[la_st1["Symbol"].isin(list(set(total_list) - set(new_list)))].loc[:,
       ["Symbol", "Name", "daily_tv_x", "daily_amt_x"]]
dff2 = dff2.sort_values(by=["daily_tv_x"], ascending=False)
dff2.to_csv("E:\\预测中证调出.csv", encoding = "GBK")
dff3 = la_st1[la_st1["Symbol"].isin(list(set(total_list1) - set(total_list)))].loc[:,
       ["Symbol", "Name", "daily_tv", "daily_amt"]]
#dff3.to_csv("E:\\实际中证调入.csv", encoding = "GBK")
dff4 = la_st1[la_st1["Symbol"].isin(list(set(total_list) - set(total_list1)))].loc[:,
       ["Symbol", "Name", "daily_tv", "daily_amt"]]
#dff4.to_csv("E:\\实际中证调出.csv", encoding = "GBK")
n = len(list(set(total_list1) & set(new_list)))
print('第一轮预测后与实际值相同的股票数为: %d' % (n))
n1 = len(list(set(total_list2) & set(new_list)))
n2 = len(list(set(total_list3) & set(new_list)))
print('第一轮预测后不变的股票与实际值相同的股票数为: %d' % (n1))
print('第一轮预测后新入的股票与实际值相同的股票数为: %d' % (n2))
tni = list(set(new_list) - set(total_list))
print(list(set(tni) & set(total_list3)))
print(l1[list(set(tni) - set(total_list3))].describe())
print(l1[list(set(total_list3) - set(tni))].describe())
print(l2[list(set(tni) - set(total_list3))].describe())
print(l2[list(set(total_list3) - set(tni))].describe())

