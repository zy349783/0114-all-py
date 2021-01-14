import numpy as np
import pandas as pd
import glob

def clean(x):
    cl_ean = x["ListDays"] != 0
    x = x[cl_ean]
    cl_ean1 = x["trade_stats"] == 1
    x = x[cl_ean1]
    return x.iloc[:,[0,1,2,3,5,6,7,8,9,10,11,12,13]]

path = r'F:\Download\StockFactors'
all_files = glob.glob(path + "/*.csv")
dd = clean(pd.read_csv(all_files[0], encoding="GBK"))
for i in range(1, len(all_files)):
    dn = clean(pd.read_csv(all_files[i], encoding="GBK"))
    dd = pd.concat([dd, dn], axis = 0, ignore_index = True)
dd = dd.sort_values(by=["Symbol", "Date"])
dd.to_csv('F:\\all_stock_2020.csv', encoding="GBK")



# path = r'E:\stockDaily'
# all_files = glob.glob(path + "/*.csv")
# dd = pd.read_csv(all_files[0], encoding = "GBK").loc[:, ["Symbol", "Date", "trade_stats"]]
# for i in range(1, len(all_files)):
#     dn = pd.read_csv(all_files[i], encoding = "GBK").loc[:, ["Symbol", "Date", "trade_stats"]]
#     dd = pd.concat([dd, dn], axis = 0, ignore_index = True)
# dd = dd.sort_values(by=["Symbol", "Date"])
# dd.to_csv('E:\\trade_status.csv', encoding = "GBK")

