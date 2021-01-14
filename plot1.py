import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import Formatter

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

df1 = pd.read_csv("E:\\benchmark_pnl.csv", encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv("E:\\Drop1d_pnl.csv", encoding="utf-8").iloc[:, 1:]
df3 = pd.read_csv("E:\\Drop5days_pnl.csv", encoding="utf-8").iloc[:, 1:]
df4 = pd.read_csv("E:\\Drop10days_pnl.csv", encoding="utf-8").iloc[:, 1:]
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(np.arange(len(df1)), df2["P&L"] - df1["P&L"], color='blue', alpha=2, linewidth=1, linestyle='-', marker='.',
        markersize=2, label='P&L difference between benchmark and drop one day early')
ax.plot(np.arange(len(df1)), df3["P&L"] - df1["P&L"], color='green', alpha=2, linewidth=1, linestyle='-', marker='.',
        markersize=2, label='P&L difference between benchmark and drop five days early')
ax.plot(np.arange(len(df1)), df4["P&L"] - df1["P&L"], color='red', alpha=2, linewidth=1, linestyle='-', marker='.',
        markersize=2, label='P&L difference between benchmark and drop ten days early')
print("Benchmark Total P&L: " + str(df1["P&L"].iloc[-1]))
print("Drop stocks 1 trading day earlier P&L: " + str(df2["P&L"].iloc[-1]))
print("Drop stocks 5 trading days earlier P&L: " + str(df3["P&L"].iloc[-1]))
print("Drop stocks 10 trading days earlier P&L: " + str(df4["P&L"].iloc[-1]))





ax.set_xlabel('')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(MyFormatter(df1["Date"].values, '%Y%m%d'))
ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
ax.set_title("Difference in P&L for market-value-weighted stocks ")
ax.grid()
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(np.arange(len(df1)), df3["P&L"] - df1["P&L"], color='blue', alpha=2, linewidth=1, linestyle='-', marker='.',
        markersize=2, label='P&L difference')
ax.set_xlabel('')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(MyFormatter(df1["Date"].values, '%Y%m%d'))
ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
ax.set_title("Difference in P&L between benchmark and Drop five days early")
ax.grid()
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(np.arange(len(df1)), df4["P&L"] - df1["P&L"], color='blue', alpha=2, linewidth=1, linestyle='-', marker='.',
        markersize=2, label='P&L difference')
ax.set_xlabel('')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(MyFormatter(df1["Date"].values, '%Y%m%d'))
ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
ax.set_title("Difference in P&L between benchmark and Drop ten days early")
ax.grid()
plt.show()