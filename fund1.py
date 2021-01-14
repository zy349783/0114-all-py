import tushare as ts
import pandas as pd
import numpy as np
import re
import glob
import pickle
from twisted.internet import task, reactor
from datetime import datetime
import schedule
import time

token = '8722319c0d4b316258408cf41d9e781fa5e3cbb4695e4e2e492636e8'
ts.set_token(token)
pro = ts.pro_api()

df = pd.DataFrame()
stocks = pd.read_csv("E:\\all_stock_2010_2019.csv", encoding="GBK")
date = stocks["Date"][(stocks["Date"] >= 20150930) & (stocks["Date"] <= 20171231)].unique()
for i in date:
    df1 = pro.fund_nav(end_date=str(i))
    df = pd.concat([df, df1])
    print(df)
df.to_csv("E:\\公募基金净值数据1.csv", encoding="utf-8")
