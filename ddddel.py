import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import Formatter
from bisect import bisect_left, bisect_right

F1 = open("C:\\Users\\win\\Downloads\\Fundholding1.pkl", 'rb')
F2 = open("C:\\Users\\win\\Downloads\\marketHolderShare.pkl", 'rb')
F3 = open("E:\\公募基金持仓数据1.pkl", 'rb')
F5 = open("E:\\公募基金持仓数据2.pkl", 'rb')
F4 = open("C:\\Users\\win\\Downloads\\inst_holding_20190930.pkl", 'rb')
ty_pe = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\fund_type.csv', encoding="GBK")
ty_pe["证券代码"] = ty_pe["证券代码"].apply(lambda x: x.split('.')[1] + x.split('.')[0])
industry = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\行业.csv', encoding="GBK")
industry["证券代码1"] = industry["证券代码"].apply(lambda x: x.split('.')[0])
industry["证券代码2"] = industry["证券代码"].apply(lambda x: x.split('.')[1] + x.split('.')[0])
CSI500 = pd.read_csv("C:\\Users\\win\\Downloads\\index_comp_SH000905.csv", encoding="utf-8")
df1 = pickle.load(F1)
df2 = pickle.load(F2)
df3 = pickle.load(F3)
df5 = pickle.load(F5)
df4 = pickle.load(F4)
df3 = pd.concat([df3, df5])
df3["end_date"] = df3["end_date"].astype(int)
df3["ts_code"] = df3["ts_code"].apply(lambda x: x.split('.')[1] + x.split('.')[0])
mv = pd.read_csv('E:\\all_stock_2010_2019.csv', encoding="GBK").iloc[:, 1:]
sw = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\申万一级行业指数.csv', encoding="GBK")

df1 = pd.merge(df1, industry, left_on="StockID", right_on="证券代码1")
df1 = df1[(df1["TargetDate"] >= min(mv["Date"])) & (df1["TargetDate"] < 20191231)]
date = np.sort(df1["TargetDate"].unique())[1:][::2]
fb = df1[df1["TargetDate"].isin(date)]
fb1 = fb.groupby(["TargetDate", "所属申万行业指数"])["HoldingValue"].sum().reset_index()
fb2 = fb1.groupby("TargetDate")["HoldingValue"].sum().reset_index()
fb1 = pd.merge(fb1, fb2, on="TargetDate")
fb1["perc"] = fb1["HoldingValue_x"] / fb1["HoldingValue_y"]
date1 = CSI500["Date"].values
date2 = []
for i in date:
    date2.append(date1[bisect_right(date1, i)-1])
CSI500 = CSI500[CSI500["Date"].isin(date2)]
dff = pd.DataFrame(CSI500.iloc[:, 1:].values.T, index=CSI500.columns[1:], columns=CSI500.iloc[:, 0].values).reset_index()
dff = pd.merge(dff, industry, left_on="index", right_on="证券代码2")
ind = pd.DataFrame()
for i in date2:
    ind1 = dff.loc[:, [i, "所属申万行业指数"]][dff[i]!=0].groupby("所属申万行业指数")[i].sum().reset_index()
    ind1["TargetDate"] = date[date2.index(i)]
    ind1 = ind1.rename(columns={i: "perc1"})
    ind = pd.concat([ind, ind1])
re = pd.merge(fb1, ind, on=["TargetDate", "所属申万行业指数"], how="inner")
re["alloc"] = re["perc"]*100 - re["perc1"]
sw = pd.DataFrame(sw.iloc[:, 2:].values.T, index=sw.columns[2:], columns=sw["证券名称"]).reset_index()
sw["index"] = sw["index"].astype(int)

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
pdd = pd.pivot_table(re.loc[:, ["TargetDate", "所属申万行业指数", "alloc"]], index=["TargetDate"], columns=["所属申万行业指数"])
pdd.columns = pdd.columns.levels[1]
import seaborn as sns
fig = plt.figure()
f, ax= plt.subplots(figsize = (14, 10))
sns_plot = sns.heatmap(pdd, annot=False, cmap="RdBu_r",center=0, linewidths=0.05, ax=ax)
plt.tick_params(labelsize=10)
ax.set_title("行业超配/低配情况热力图")
plt.show()