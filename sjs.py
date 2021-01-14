import tushare as ts
import pandas as pd
import re
import numpy as np

token='8722319c0d4b316258408cf41d9e781fa5e3cbb4695e4e2e492636e8'
ts.set_token(token)
pro = ts.pro_api()

df = pd.read_csv("E:\\all_stock.csv", encoding="GBK")
list = df['Symbol'].unique()
ndf = pd.DataFrame()
ndf["ID"] = pd.Series(list)
ndf["start_date"] = df.groupby("Symbol")["Date"].first().reset_index()["Date"]
ndf["end_date"] = df.groupby("Symbol")["Date"].last().reset_index()["Date"]
ndf = ndf[ndf["ID"].str[:2] == "SZ"]
da_te = pd.Series(np.sort(df["Date"].unique()))
for i in range(413, len(ndf)):
    ID = ndf.iloc[i, 0]
    ID = re.split('(\d+)', ID)[1] + '.' + re.split('(\d+)', ID)[0]
    d1 = str(ndf.iloc[i, 1])
    d2 = str(ndf.iloc[i, 2])
    d3 = str(da_te[da_te.index[int(d1) == da_te].tolist()[0] + 30])
    gg1 = pro.anns(ts_code=ID, start_date=d1, end_date=d3)
    if len(gg1) == 0:
        d3 = str(da_te[da_te.index[int(d1) == da_te].tolist()[0] + 60])
        gg1 = pro.anns(ts_code=ID, start_date=d1, end_date=d3)
        if len(gg1) == 0:
            d3 = str(da_te[da_te.index[int(d1) == da_te].tolist()[0] + 90])
            gg1 = pro.anns(ts_code=ID, start_date=d1, end_date=d3)
            if len(gg1) == 0:
                d3 = str(da_te[da_te.index[int(d1) == da_te].tolist()[0] + 120])
                gg1 = pro.anns(ts_code=ID, start_date=d1, end_date=d3)
    print(gg1["ann_date"].iloc[-1])
    gg = pro.anns(ts_code=ID, start_date=d1, end_date=d2)
    while gg["ann_date"].iloc[-1] != gg1["ann_date"].iloc[-1]:
        if len(da_te.index[int(gg["ann_date"].iloc[-1]) == da_te].tolist()) == 0:
            d2 = str(da_te[da_te.index[int(gg["ann_date"].iloc[-1]) < da_te].tolist()[0] - 1])
        else:
            d2 = str(da_te[da_te.index[int(gg["ann_date"].iloc[-1]) == da_te].tolist()[0] - 1])
        gg2 = pro.anns(ts_code=ID, start_date=d1, end_date=d2)
        gg = pd.concat([gg, gg2])
        print(gg)
    gg = gg.reset_index().iloc[:, 1:]
    gg.to_csv("E:\\SZ announcement\\" + ID + ".csv", encoding="utf-8")
    print(gg)
print(ndf)

