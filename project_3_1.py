import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import timedelta
import calendar
import seaborn as sns

# 1. Get list of stocks new_in, old_out, old_in at the time of underlying changing
csi_index = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv', encoding = "utf-8")
beta1 = pd.read_csv('E:\\beta1.csv', encoding = "utf-8").iloc[:,1:]
beta2 = pd.read_csv('E:\\beta2.csv', encoding = "utf-8").iloc[:,1:]
beta1["Date"] = beta1["Date"].apply(lambda x: str(x))
beta1["Date"] = pd.to_datetime(beta1["Date"])
beta2["Date"] = beta2["Date"].apply(lambda x: str(x))
beta2["Date"] = pd.to_datetime(beta2["Date"])
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
# Take the event on 2019.06.17 as an example, first get the time period
event_date = time_df["Effect Day"].iloc[-1]
time_period = da_te[da_te[da_te == event_date].index[0]-20:da_te[da_te == event_date].index[0]+21]
# calculate time dependent CAR
stocks_new_in = new_in.iloc[:,-1]
stocks_old_out = old_out.iloc[:,-1] #SZ000418停牌 SZ000916
#stocks_old_out = stocks_old_out.drop(stocks_old_out[stocks_old_out == 'SZ000418'].index[0])
#stocks_old_out = stocks_old_out.drop(stocks_old_out[stocks_old_out == 'SZ000979'].index[0])
stocks_old_in = old_in.iloc[:,-1]

def cal_CAR(stocks):
    df = pd.DataFrame()
    df["Date"] = time_period
    df.reset_index(drop=True, inplace=True)
    CAR =[]
    T = []
    n = int((len(time_period)-1)/2)
    for i in stocks:
        data1 = beta2[beta2["Symbol"] == i]
        data1 = data1[(data1['Date'] >= time_period.iloc[0]) & (data1['Date'] <= time_period.iloc[-1])]["alpha"]
        print(i)
        print(len(data1))
        df[i] = pd.Series(data1.values, index=df.index)
    for i in range(len(time_period)):
        s_um = 0
        for j in range(len(stocks)):
            if i < n:
                car = np.sum(df.iloc[i:n, j+1])
            else:
                car = np.sum(df.iloc[n:i+1, j+1])
            s_um = s_um + car
        CAR.append(s_um / len(stocks))
        T.append(np.mean(df.iloc[i, 1:])*np.sqrt(len(stocks))/np.std(df.iloc[i, 1:]))
    df["t"] = pd.Series(T, index=df.index)
    df["CAR"] = pd.Series(CAR, index=df.index)
    return df

df1 = cal_CAR(stocks_old_out)
t = df1["t"]
CAR = df1["CAR"]
v1 = np.insert(CAR.values, 20, 0)
v2 = np.insert(t.values, 20, 0)
x = range(-20, 22)

# 4. Plot
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

sns.set()
fig, ax1 = plt.subplots()
ax1.set_ylabel('t value', fontname="Arial", fontsize=6)
ax1.bar(x, v2, color='salmon', alpha=0.8)
ax1.tick_params('y')
ax1.tick_params(labelsize=6)
ax1.axhline(y=1.96, color='green', linestyle=(0,(1,1)), alpha=0.6, linewidth=1)
ax1.axhline(y=-1.96, color='green', linestyle=(0,(1,1)), alpha=0.6, linewidth=1)
ax1.axhline(y=0, color='salmon', linestyle=(0,(1,1)), alpha=0.6, linewidth=1)
ax1.axvline(x=0, color='salmon', linestyle=(0,(1,1)), alpha=0.6, linewidth=1)
ax2 = ax1.twinx()
ax2.set_ylabel('Cumulative Abnormal Returns', fontname="Arial", fontsize=6)
ax2.plot(x, v1, marker='.', color='blue', alpha=0.5, linewidth=1, markersize=2)
ax2.tick_params('y')
ax2.tick_params(labelsize=6)

plt.title("Cumulative Average alpha before/after Implementation on 2018.12.17", fontname="Arial", fontsize=8)
ax1.grid(True)
ax2.grid(None)
align_yaxis_np(ax1, ax2)
fig.tight_layout()
plt.show()



