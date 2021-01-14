import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta
import calendar
import seaborn as sns
import glob
from datetime import datetime

sto_ck = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]

def volume_plot(start_date, end_date):
    test = sto_ck[(sto_ck["Date"] >= start_date) & (sto_ck["Date"] <= end_date)]
    df = pd.DataFrame()
    df["Date"] = test["Date"].unique()
    df["MarketValue"] = test.groupby("Date")["MarketValue"].sum().reset_index()["MarketValue"]
    df["TradingVolume"] = test.groupby("Date")["volume"].sum().reset_index()["volume"]
    df["TradingVolume"] = df["TradingVolume"]/100000000
    df["amt"] = test.groupby("Date")["amt"].sum().reset_index()["amt"]
    df["TUN"] = df["amt"] / (df["MarketValue"] * 10000)
    df["Date"] = df["Date"].apply(lambda x: str(x))
    date_time = pd.to_datetime(df["Date"])
    df["TradingVolume"] = df["TradingVolume"].rolling(10).mean()
    df["TUN"] = df["TUN"].rolling(10).mean()

    fig, ax = plt.subplots()
    ax.plot(date_time, df["TradingVolume"])
    plt.title("Trading Volume from " + str(start_date) + " to " + str(end_date))
    ax.set_ylabel('Trading Volume (billion yuan)')
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(date_time, df["TUN"])
    plt.title("Turnover rates from " + str(start_date) + " to " + str(end_date))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    ax.grid(True)
    plt.show()

volume_plot(20170101, 20191128)