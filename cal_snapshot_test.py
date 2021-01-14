import numpy as np
import pandas as pd
import pickle

# df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
# df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
# df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
# df2 = df2.sort_values("sequenceNo")
# df3 = df3[df3["exchId"] == 2]
# df3 = df3.sort_values("sequenceNo")
# df = pd.DataFrame()



F1 = open("C:\\Users\\win\\Downloads\\forZhenyu\\mdLog.pkl",'rb')
F2 = open("C:\\Users\\win\\Downloads\\forZhenyu\\orderLog.pkl",'rb')
F3 = open("C:\\Users\\win\\Downloads\\forZhenyu\\tradeLog.pkl",'rb')
df1 = pickle.load(F1).iloc[:, 1:]
df2 = pickle.load(F2).iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pickle.load(F3).iloc[:, [1, 2, 5, 8, 9, 12, 13, 14, 15, 16]]
df2 = df2.sort_values("sequenceNo")
df3 = df3[df3["exchId"] == 2]
df3 = df3.sort_values("sequenceNo")
df = pd.DataFrame()
print("get data now")

for i in df2["SecurityID"].unique():
    dff = pd.read_csv("E:\\Snapshot Data1\\" + str(i) + ".csv", encoding="utf-8").iloc[:, 1:]
    df = pd.concat([dff, df])
columns = ["StockID", "cum_volume", "cum_amount", "close", "bid1p", "bid2p", "bid3p", "bid4p", "bid5p",
               "bid1q", "bid2q", "bid3q", "bid4q", "bid5q", "ask1p", "ask2p", "ask3p", "ask4p", "ask5p", "ask1q",
               "ask2q", "ask3q", "ask4q", "ask5q", "openPrice"]

dfr = df1[df1["StockID"].isin(df2["SecurityID"].unique())]
dfr["time"] = dfr["time"].apply(lambda x: int((x.replace(':', "")).replace(".", "")))
dfr = dfr[((dfr["time"] >= 93003000) & (dfr["time"] <= 112957000) & (dfr["source"] == 4)) |
          ((dfr["time"] >= 130003000) & (dfr["time"] <= 145657000) & (dfr["source"] == 4))]
dfr = dfr.sort_values(by=["StockID", "sequenceNo"])
df = df.sort_values(by=["StockID", "sequenceNo"])
df.to_pickle(r"E:\\Snapshot Data1\\result.pkl")
if pd.merge(dfr, df, left_on=columns, right_on=columns, how="outer").dropna(subset=["sequenceNo_x"]) \
    ["sequenceNo_y"].isna().any() == False:
    print("all stocks snapshot data correct")
else:
    print("not all snapshot data correct")



