import numpy as np
import pandas as pd

df1 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_SZ_20191202_0834.csv', encoding="utf-8").iloc[:, 1:]
df2 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 2, 5, 7, 8, 9, 10, 12, 13]]
df3 = pd.read_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_20191202_0834.csv', encoding="utf-8").iloc[:, [1, 5, 8, 9, 12, 13, 14, 15, 16]]

test1 = df2[(df2["SecurityID"] == 166) | (df2["SecurityID"] == 2442) | (df2["SecurityID"] == 631)]
test1["OrderType"] = test1["OrderType"].apply(lambda x: str(x))
trade1 = df3[(df3["SecurityID"] == 166) | (df3["SecurityID"] == 2442) | (df3["SecurityID"] == 631)]
re1 = df1[(df1["StockID"] == 166) | (df1["StockID"] == 2442) | (df1["StockID"] == 631)]

test1.to_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdLog_test.csv', encoding="utf-8")
trade1.to_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdOrderLog_test.csv', encoding="utf-8")
re1.to_csv('E:\\forZhenyu\\logs_20191202_zs_92_01_day_data\\mdTradeLog_test.csv', encoding="utf-8")