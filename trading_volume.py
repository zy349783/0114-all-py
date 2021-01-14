import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns
from matplotlib.ticker import Formatter
from matplotlib import pyplot as plt



class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos"""
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return pd.to_datetime(self.dates[ind], format="%Y%m%d").strftime(self.fmt)

sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
sto_ck = sto_ck[sto_ck["Date"] >= 20151001]
df = pd.DataFrame()
for i in sorted(sto_ck["Date"].unique()):
    tv = sto_ck[sto_ck["Date"] == i]["volume"].sum()
    df = df.append(pd.DataFrame({"Date": i, "volume": tv}, index=[0]), ignore_index=True)

list1 = [20160205, 20160930, 20170126, 20170929, 20180214, 20180928, 20190201, 20190930]
list2 = []
for i in list1:
    list2.append(df.index[df["Date"] == i].values[0])
test = pd.DataFrame()
df["diff"] = df["volume"] - df["volume"].shift(1)
df["rolling"] = df["volume"] / df["volume"].rolling(30).mean()
for i in list2:
    df1 = df.loc[i-60: i, :]
    test = pd.concat([test, pd.DataFrame(df1["rolling"].values)], axis=1)

# M = test.mean(axis=1)
M = test.iloc[:, -1]


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

x = range(-61, 0, 1)
sns.set()
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

# ax1.set_ylabel('cumulative trading volume diff', fontname="Arial", fontsize=8)
# ax1.plot(x, c_um, marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
# ax1.axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax1.set_ylabel('DailyVolume/RollingMeanVolume', fontname="Arial", fontsize=8)
ax1.bar(x, M, color='salmon', alpha=0.8)
# ax2.scatter(x, M1, color='red', alpha=0.8, s=4)
ax1.tick_params('y')
ax1.tick_params(labelsize=8)
plt.title("DailyVolume/RollingMeanVolume before 2019.10.1", fontname="Arial", fontsize=10)
ax1.axhline(y=1, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax1.grid(True)


fig.tight_layout()
plt.show()