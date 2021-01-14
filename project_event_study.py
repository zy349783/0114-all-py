import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import timedelta
import calendar
import seaborn as sns


csi_index = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv', encoding="utf-8")
csi_index_300 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000300.csv', encoding="utf-8")
csi_index_1000 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000852.csv', encoding="utf-8")
csi_index_all = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000985.csv', encoding="utf-8")
csi_index["Date"] = csi_index["Date"].apply(lambda x: str(x))
csi_index_300["Date"] = csi_index_300["Date"].apply(lambda x: str(x))
csi_index_1000["Date"] = csi_index_1000["Date"].apply(lambda x: str(x))
csi_index_all["Date"] = csi_index_all["Date"].apply(lambda x: str(x))
da_te = pd.to_datetime(csi_index["Date"])
sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:,1:]
sto_ck["returns"] = sto_ck.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck["Date"] = sto_ck["Date"].apply(lambda x: str(x))
sto_ck["Date"] = pd.to_datetime(sto_ck["Date"])
xx = []
dd1 = []
dd2 = []
xx.append(csi_index.columns[csi_index.iloc[0,:]!=0])
new_in = pd.DataFrame()
old_out = pd.DataFrame()
old_in = pd.DataFrame()
old_out_hs300 = pd.DataFrame(index=np.arange(50))
old_out_zz1000 = pd.DataFrame(index=np.arange(50))
old_out_zzall = pd.DataFrame(index=np.arange(50))
old_out_else = pd.DataFrame(index=np.arange(50))
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

    event_date = time_df[day].iloc[len(time_df)-1]
    stocks = dataframe.iloc[:, -1][~dataframe.iloc[:, -1].isnull()]
    for j in stocks:
        st_r = str(event_date)[0:4] + str(event_date)[5:7] + str(event_date)[8:10]
        event_date1 = da_te[da_te.index[da_te == time_df["Effect Day"].iloc[len(time_df)-1]][0]-1]
        st_r1 = str(event_date1)[0:4] + str(event_date1)[5:7] + str(event_date1)[8:10]
        min_data1 = pd.read_parquet('E:\\alpha\\alpha_min_' + st_r1 + '.parquet', engine='pyarrow')
        min_data2 = pd.read_parquet('E:\\alpha\\alpha_min_' + st_r + '.parquet', engine='pyarrow')
        close2open = pd.read_parquet('E:\\alpha_LD1close2open.parquet', engine='pyarrow')
        open21min = pd.read_parquet('E:\\alpha_open21min.parquet', engine='pyarrow')
        tp = min_data1[min_data1["ID"] == j]["alpha"].dropna()[-20:]
        tp = tp.reset_index()["alpha"]
        tp = tp.append(close2open[(close2open["Date"] == int(st_r)) & (close2open["ID"] == j)]["alphaLD1close_open"],
                  ignore_index=True)
        tp = tp.append(open21min[(open21min["Date"] == int(st_r)) & (open21min["ID"] == j)]["alpha_open21min"],
                  ignore_index=True)
        tp = tp.append(min_data2[min_data2["ID"] == j]["alpha"].dropna()[1:20], ignore_index=True)
        tp = pd.DataFrame(tp, columns=["alpha_" + j])
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

cal("Effect Day", old_out_hs300)
cal("Effect Day", old_out_zz1000)
cal("Effect Day", old_out_zzall)
cal("Effect Day", old_out_else)