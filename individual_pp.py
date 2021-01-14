import pandas as pd
import numpy as np

sz = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 2\\SZInvestor.csv', encoding = "utf-8")
ss = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 2\\SHInvestor.csv', encoding = "utf-8")
sz['StockID'] = 'SZ' + sz['StockID'].map(lambda x: f'{x:0>6}')
ss['StockID'] = 'SH' + ss['StockID'].astype(str)
ss = ss.rename({'r_ind':'ind_pc'}, axis=1)
sz['ind_pc'] = sz['type1'] + sz['type2']
data = pd.concat([ss.loc[:,["StockID","ind_pc"]],sz.loc[:,["StockID","ind_pc"]]], axis=0)
data.to_csv('E:\\investors.csv', encoding="utf-8")