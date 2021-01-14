import pandas as pd
import matplotlib as mpl
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# 1. Get list of stocks new_in, old_out at the time of underlying changing
csi_index = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv', encoding = "utf-8")
sto_ck = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]
sto_ck["returns"] = sto_ck.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck["Date"] = sto_ck["Date"].apply(lambda x: str(x))
sto_ck["Date"] = pd.to_datetime(sto_ck["Date"])
csi_index["Date"] = csi_index["Date"].apply(lambda x: str(x))
da_te = pd.to_datetime(csi_index["Date"])
xx = []
dd1 = []
dd2 = []
xx.append(csi_index.columns[csi_index.iloc[0,:]!=0])
new_in = pd.DataFrame()
old_out = pd.DataFrame()
old_in = pd.DataFrame()
for i in range(1,len(csi_index)):
    xx.append(csi_index.columns[csi_index.iloc[i,:]!=0])
    #if (xx[i-1][1:] == xx[i][1:]).all() == False:
        #tt = [x for x in xx[i][1:] if x not in xx[i-1][1:]]
        #t_ry[str(csi_index.iloc[i,0])] = pd.Series(tt)
    new_in[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i][1:]) - set(xx[i - 1][1:])))
    old_out[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i - 1][1:]) - set(xx[i][1:])))
    old_in[csi_index.iloc[i, 0]] = pd.Series(list(set(xx[i - 1][1:]) & set(xx[i][1:])))

new_in = new_in.dropna(axis=1, how="any")
old_out = old_out.dropna(axis=1, how="any")
old_in = old_in[new_in.columns]
old_in = old_in.dropna(axis=0, how="all")
print(da_te)
print(new_in)
print(old_out)
print(old_in)

# 2. Get the exact effect day and notice day from 2010 to 2019
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
    # when Friday is holiday, we need to look back two days to match trading days
    elif (da_te == second_friday - timedelta(days=2)).any():
        date1 = da_te[da_te[da_te == second_friday - timedelta(days=2)].index[0] + 1]
        date2 = date1 - timedelta(days=14)
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
# Take the event on 2019.06.17 as an example, first get the time period
def estimate_day(t):
    if t.weekday() == 6 or t.weekday() == 7:
        return t + timedelta(days=8-t.weekday())
    else:
        return t
event_date = time_df["Effect Day"].iloc[-1]
t1 = event_date - timedelta(days=20)
t2 = event_date + timedelta(days=20)
t1 = estimate_day(t1)
t2 = estimate_day(t2)
time_period = da_te[da_te[da_te == t1].index[0]:da_te[da_te == t2].index[0]]
# calculate time dependent CAR
stocks_new_in = new_in.iloc[:,-1]
stocks_old_out = old_out.iloc[:,-1]
stocks_old_in = old_in.iloc[:,-1]

#p1 = pd.Index(sto_ck[sto_ck["Symbol"] == "SZ300009"]["Date"]).get_loc(time_period.iloc[0])
#p2 = pd.Index(sto_ck[sto_ck["Symbol"] == "SZ300009"]["Date"]).get_loc(time_period.iloc[-1])
def cal_CAR(stocks):
    df = pd.DataFrame()
    abnormal_returns = []
    for i in stocks:
        data1 = sto_ck[sto_ck["Symbol"] == i]
        data1 = data1[(data1['Date'] >= time_period.iloc[0]) & (data1['Date'] <= time_period.iloc[-1])]["returns"]
        df[i] = pd.Series(data1.values)
    for i in range(len(time_period)):
        s_um = 0
        k = 0
        for j in range(len(stocks)):
            #err_or = df.iloc[i, j] - np.nanmean(df.iloc[:i+1, j])
            err_or = df.iloc[i, j] - np.nanmean(df.iloc[:, j])
            if np.isnan(err_or):
                err_or = 0
                k = k + 1
            s_um = s_um + err_or
        abnormal_returns.append(s_um / len(stocks))
    df["AR"] = pd.Series(abnormal_returns, index=df.index)
    df["CAR"] = pd.Series(np.cumsum(abnormal_returns), index=df.index)
    return df

def cal_t(dataframe):
    t = []
    for i in range(len(time_period)):
        #sig_ma = np.std(dataframe.ix[:i+1, "AR"])
        sig_ma = np.std(dataframe.loc[:, "AR"])
        t.append(dataframe.ix[i, "CAR"] / (np.sqrt(i+1)*sig_ma))
    dataframe["t-statistics"] = pd.Series(t, index=dataframe.index)

new_in = pd.DataFrame()
old_out = pd.DataFrame()
old_in = pd.DataFrame()
new_in = cal_CAR(stocks_new_in)
new_in = new_in.set_index(time_period)
old_out = cal_CAR(stocks_old_out)
old_out = old_out.set_index(time_period)
old_in = cal_CAR(stocks_old_in)
old_in = old_in.set_index(time_period)
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)
plt.xticks(rotation=90)
plt.plot(new_in.index, new_in["CAR"].values, 'r-', label="new_in stocks")
plt.plot(new_in.index.values[13], new_in["CAR"].values[13], marker='o', markersize=3, color='red')
plt.plot(old_out.index, old_out["CAR"].values, 'b-', label="old_out stocks")
plt.plot(new_in.index.values[13], old_out["CAR"].values[13], marker='o', markersize=3, color='blue')
plt.plot(old_in.index, old_in["CAR"].values, 'g-', label="old_in stocks")
plt.plot(new_in.index.values[13], old_in["CAR"].values[13], marker='o', markersize=3, color='green')
plt.xlabel("Date")
plt.ylabel("CAR")
plt.title("Plot of Cumulative Constant-Mean-Return Model Mean Abnormal Return for CSI500 adjustment implementation on 2019.06.17")
plt.legend()
plt.show()

cal_t(new_in)
cal_t(old_out)
cal_t(old_in)
new_in.to_csv('E:\\new_in.csv', encoding="utf-8")
old_out.to_csv('E:\\old_out.csv', encoding="utf-8")
old_in.to_csv('E:\\old_in.csv', encoding="utf-8")


total_list = pd.concat([new_in.iloc[:,-2],old_in.iloc[:,-2]])
total_list1 = pd.concat([new_in.iloc[:,-1],old_in.iloc[:,-1]])
total_list2 = old_in.iloc[:,-1]
total_list3 = new_in.iloc[:,-1]
data1 = sto_ck[(sto_ck['Date'] >= "20180502") & (sto_ck['Date'] <= "20190430")].loc[:,["Date","Symbol","MarketValue","amt"]]
l1 = data1.groupby("Symbol")["MarketValue"].mean()
l2 = data1.groupby("Symbol")["amt"].mean()
data1 = data1.groupby("Symbol").last().reset_index()
data1["daily_mktv"] = l1.values
data1["daily_amt"] = l2.values
data1["mktv_rank"] = data1["daily_mktv"].rank(ascending=False)
data1["amt_rank"] = data1["daily_amt"].rank(ascending=False)
hs = data1[data1["amt_rank"] < len(data1)/2].sort_values(by=["mktv_rank"])[:300]
data1.drop(hs.index, axis=0, inplace=True)
data1["CSI500"] = np.nan
for i in range(len(total_list)):
    data1.loc[data1["Symbol"] == total_list.iloc[i],["CSI500"]] = 1
data1["mktv_rank"] = data1["daily_mktv"].rank(ascending=False)
data1["amt_rank"] = data1["daily_amt"].rank(ascending=False)
data1 = pd.concat([data1[(data1["CSI500"] == 1) & (data1["amt_rank"] < len(data1)*0.9)], data1[np.isnan(data1["CSI500"])]])
data1 = data1.sort_values(by=['mktv_rank'])
new_list = pd.concat([data1[data1["CSI500"]==1][:450], data1[data1["CSI500"]!=1][:50]])["Symbol"]
n = len(list(set(total_list1) & set(new_list)))
print('第一轮预测后与实际值相同的股票数为: %d' % (n))
n1 = len(list(set(total_list2) & set(new_list)))
n2 = len(list(set(total_list3) & set(new_list)))
print('第一轮预测后未变的股票与实际值相同的股票数为: %d' % (n1))
print('第一轮预测后新入的股票与实际值相同的股票数为: %d' % (n2))


