import pickle
import pandas as pd
import numpy as np

import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

F1 = open("C:\\Users\\win\\Downloads\\Fundholding.pkl", 'rb')
F2 = open("C:\\Users\\win\\Downloads\\marketHolderShare.pkl", 'rb')
F3 = open("C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\公募基金持仓数据.pkl", 'rb')
ty_pe = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\fund_type.csv', encoding="GBK")
ty_pe["证券代码"] = ty_pe["证券代码"].apply(lambda x: x.split('.')[1] + x.split('.')[0])
df1 = pickle.load(F1)
df2 = pickle.load(F2)
df3 = pickle.load(F3)
mv = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
df1 = df1[df1["TargetDate"] >= min(mv["Date"])]
# fb = df1.groupby(["TargetDate", "Sector"])["HoldingValue"].sum().reset_index()
fb = df1.groupby(["TargetDate", "Sector"])["ratio"].sum().reset_index()
fb = fb[fb["TargetDate"] >= 20180000]
fb = fb[(fb["Sector"]!='0') & (fb["Sector"]!="")]
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
f, ax = plt.subplots(figsize = (6,15))
sns.barplot(x="TargetDate", hue="Sector", y="ratio", data=fb)
ax.legend(ncol=2, loc='upper left')
plt.show()

