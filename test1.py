import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import Formatter

df1 = pd.read_csv("E:\\stock balance4\\20190507.csv", encoding="utf-8")
df2 = pd.read_csv("E:\\stock balance4\\20190506.csv", encoding="utf-8")
df3 = pd.read_csv("E:\\stock balance1\\20190507.csv", encoding="utf-8")
df4 = pd.read_csv("E:\\stock balance1\\20190506.csv", encoding="utf-8")
data = pd.read_csv('E:\\all_stock_2019.csv', encoding="GBK").iloc[:, 1:]
data.loc[data["trade_stats"] == 0, "ret"] = 0
data["returns"] = data["ret"]/100
ST = pd.read_csv("E:\\跌停统计1.csv", encoding="GBK")
ST.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
ST = ST.sort_values(by="首日")
ST.drop_duplicates(inplace=True)
print(df1)