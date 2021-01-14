import pandas as pd

df1 = pd.read_csv("E:\\compare\\logs_20200207_zs_92_01_day_data\\mdLog_SZ_20200207_0838.csv", encoding="utf-8")
df2 = pd.read_csv("E:\\compare\\logs_20200207_zs_92_01_day_data\\mdOrderLog_20200207_0838.csv", encoding="utf-8")
df3 = pd.read_csv("E:\\compare\\logs_20200207_zs_92_01_day_data\\mdTradeLog_20200207_0838.csv", encoding="utf-8")
df4 = pd.read_csv("E:\\mbd\\raw data\\mdLog_SZ_20200207_0838.csv", encoding="utf-8")
li_st = df4["StockID"].unique()
df1 = df1[df1["StockID"].isin(li_st)]
df2 = df2[df2["SecurityID"].isin(li_st)]
df3 = df3[df3["SecurityID"].isin(li_st)]

df1.to_csv("E:\\mbd\\raw data\\mdLog_SZ_20200207_0838.csv", encoding="utf-8", index=False)
df2.to_csv("E:\\mbd\\raw data\\mdOrderLog_20200207_0838.csv", encoding="utf-8", index=False)
df3.to_csv("E:\\mbd\\raw data\\mdTradeLog_20200207_0838.csv", encoding="utf-8", index=False)