import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns

F1 = open(r'C:\\Users\\win\\Downloads\\FastFinancialReport.pkl', 'rb')
F2 = open(r'C:\\Users\\win\\Downloads\\ForcastFinancialReport.pkl', 'rb')
kb = pickle.load(F1)
yb = pickle.load(F2)
print(kb.iloc[0, :])
print(yb.iloc[0, :])
alpha = pd.read_csv('E:\\new_beta1.csv', encoding="utf-8").iloc[:, 1:]
da_te = alpha["Date"]
yb = yb[yb["预警类型"] != ""]
types = yb["预警类型"].unique()
df1 = yb[yb["预警类型"] == "预增"].loc[:, ["公布日", "StockID"]]


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

def cal2(dff):
    df = pd.DataFrame()
    st = pd.DataFrame()
    CAR = []
    T = []
    m = 0

    dataframe = pd.DataFrame()
    # dff = dff[dff["公布日"] >= 20100104]
    dff = dff[dff["公布日"] >= 20100322]
    date1 = np.sort(dff["公布日"].unique())
    nan_list = []
    nanafter_list = []
    nanret_list = []
    for i1 in date1:
        ST2 = pd.DataFrame()
        ST2[i1] = pd.Series(dff[dff["公布日"] == i1]["StockID"].values)
        ST2[i1].drop_duplicates(inplace=True)
        dataframe = pd.concat([dataframe, ST2], axis=1)

    for i in dataframe.columns:
        stocks = dataframe[i][~dataframe[i].isnull()]
        for j in stocks:
            m = m + 1
            data1 = alpha[alpha["ID"] == j]
            da_te = data1["Date"]
            event_date = i
            if len(da_te[da_te == event_date]) == 0:
                if len(da_te[da_te > event_date]) == 0:
                    print("No trade after event_date~~~~")
                    if len(data1) == 0:
                        nan_list.append([i, j])
                    else:
                        nanafter_list.append([i, j])
                    print(len(data1))
                    print(event_date)
                    print(j)
                    continue
                else:
                    event_date = da_te[da_te > event_date].iloc[0]
            time_period = da_te.loc[da_te[da_te == event_date].index[0] - 10:da_te[da_te == event_date].index[0] + 10]
            # tp = pd.DataFrame(pd.merge(tpp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            tp = pd.DataFrame(data1.loc[time_period.index.values, "stock_return"].values, columns=[m])
            # tp1 = pd.DataFrame(pd.merge(tpp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            # if (tp.isna().all().values[0]) | (tp1.isna().all().values[0]):
            if tp.isna().all().values[0]:
                print("All nan return")
                nanret_list.append([event_date, j])
                print(event_date)
                print(j)
                continue
            else:
                # tp = tp.rename(columns={'alpha': m})
                df = pd.concat([df, tp], axis=1)
    print(m)
    print(df.shape[1])
    print(len(nan_list))
    print(len(nanafter_list))
    print(len(nanret_list))
    CAR = np.cumsum(df.mean(axis=1)) * 10000
    T = df.mean(axis=1) * 10000
    M = df.median(axis=1) * 10000
    return pd.DataFrame({"CAR": CAR, "T value": T, "Median_alpha": M})

x = range(-10, 11)
sns.set()
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

r1 = cal2(df1)
mean = r1["mean"].values[0]
median = r1["median"].values[0]
prob = r1["prob"].values[0]
prob1 = r1["prob1"].values[0]
l = r1["n"].values[0]
ax1.axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
ax1.axvspan(0, 10, facecolor='green', alpha=0.5)
ax1.set_ylabel('Cumulative Return', fontname="Arial", fontsize=8)
ax1.plot(x, r1["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
ax1.tick_params('y')
ax1.tick_params(labelsize=8)
ax1.axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Return', fontname="Arial", fontsize=8)
ax2.bar(x, r1["T value"], color='salmon', alpha=0.8)
ax2.scatter(x, r1["Median_alpha"], color='red', alpha=0.8, s=4)
ax2.tick_params('y')
ax2.tick_params(labelsize=8)
plt.title("All ST stocks cumulative return before and after effect day", fontname="Arial", fontsize=10)
ax1.grid(True)
ax2.grid(None)
align_yaxis_np(ax1, ax2)

fig.tight_layout()
plt.show()