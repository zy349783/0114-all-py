import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns

F1 = open(r'C:\\Users\\win\\Downloads\\FastFinancialReport.pkl', 'rb')
F2 = open(r'C:\\Users\\win\\Downloads\\ForcastFinancialReport.pkl', 'rb')
F3 = open(r'C:\\Users\\win\\Downloads\\event_1106.pkl', 'rb')
kb = pickle.load(F1)
yb = pickle.load(F2)
bg = pickle.load(F3)
alpha = pd.read_csv('E:\\new_beta1.csv', encoding="utf-8").iloc[:, 1:]
alpha = alpha.sort_values(by=["Date", "ID"])
date1 = pd.Series(np.sort(alpha["Date"].unique()))
yb = yb[yb["预警类型"] != ""]
types = yb["预警类型"].unique()
yb["types"] = np.nan
yb.loc[yb["预警类型"].isin(["预增","预盈","减亏"]), "types"] = "Good"
yb.loc[yb["预警类型"].isin(["预减","预亏","增亏"]), "types"] = "Bad"
yb.loc[yb["预警类型"].isin(["预平","预警"]), "types"] = "Not Sure"
print(yb)

def cal(t):
    df1 = yb[yb["types"] == t].loc[:, ["公布日", "StockID"]]
    df1 = df1[df1["公布日"] >= 20100104]
    df1.sort_values(by=["公布日", "StockID"], inplace=True)
    df1 = df1.reset_index().iloc[:, 1:]
    # df1["公布日"] = pd.merge_asof(df1, alpha, left_on="公布日", right_on="Date", left_by="StockID", right_by="ID",
    #                            direction="forward")["Date"]
    # df1.dropna(inplace=True)
    date2 = df1["公布日"].unique()
    col = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
           "d17", "d18", "d19", "d20", "d21"]
    df2 = pd.DataFrame(columns=col)
    for i in date2:
        if len(date1[date1 == i] != 0):
            dd = date1.loc[date1[date1 == i].index[0] - 9:date1[date1 == i].index[0] + 11].values
            if date1[date1 == i].index[0] - 9 < 0:
                dd = np.pad(dd, (21 - len(dd), 0), 'constant')
            if date1[date1 == i].index[0] + 11 > len(date1) - 1:
                dd = np.pad(dd, (0, 21 - len(dd)), 'constant')
            df2 = df2.append(pd.Series(dd, index=col), ignore_index=True)
        else:
            nd = date1.iloc[max(date1.index[(date1 - i) < 0].values)]
            df1.loc[df1["公布日"] == i, "公布日"] = nd
            dd = date1.loc[date1[date1 == nd].index[0] - 9:date1[date1 == nd].index[0] + 11].values
            if date1[date1 == nd].index[0] - 9 < 0:
                dd = np.pad(dd, (21 - len(dd), 0), 'constant')
            if date1[date1 == nd].index[0] + 11 > len(date1) - 1:
                dd = np.pad(dd, (0, 21 - len(dd)), 'constant')
            df2 = df2.append(pd.Series(dd, index=col), ignore_index=True)
    df2 = df2[col]
    df2.drop_duplicates(inplace=True)
    df1 = pd.merge(df1, df2, left_on="公布日", right_on="d10")
    df3 = pd.DataFrame()
    for i in col:
        tp = pd.merge(df1.loc[:, ["StockID", i]], alpha.loc[:, ["ID", "Date", "alpha"]], left_on=["StockID", i],
                      right_on=["ID", "Date"], how="left")["alpha"]
        tp = pd.DataFrame(tp.values, columns=[i])
        df3 = pd.concat([df3, tp], axis=1)
    df3 = df3.dropna(how='all')
    CAR = np.cumsum(df3.mean(axis=0)) * 10000
    T = df3.mean(axis=0) * 10000
    M = df3.median(axis=0) * 10000
    num = df3.shape[0]
    return pd.DataFrame({"CAR": CAR, "T value": T, "Median_alpha": M, "number": num})

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


x = range(-10, 11)
sns.set()
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

r1 = cal("Not Sure")
ax1.axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
ax1.axvspan(0, 10, facecolor='green', alpha=0.5)
ax1.set_ylabel('Cumulative Alpha', fontname="Arial", fontsize=8)
ax1.plot(x, r1["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
ax1.tick_params('y')
ax1.tick_params(labelsize=8)
ax1.axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Alpha', fontname="Arial", fontsize=8)
ax2.bar(x, r1["T value"], color='salmon', alpha=0.8)
ax2.scatter(x, r1["Median_alpha"], color='red', alpha=0.8, s=4)
ax2.tick_params('y')
ax2.tick_params(labelsize=8)
plt.title("Not Sure Events # " + str(r1["number"].iloc[0]), fontname="Arial", fontsize=10)
ax1.grid(True)
ax2.grid(None)
align_yaxis_np(ax1, ax2)

fig.tight_layout()
plt.show()