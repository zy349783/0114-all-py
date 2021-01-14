import pickle
import pandas as pd
import statistics
import glob
import numpy as np

def clean(x):
    x1 = x.iloc[:,[0,1,2,9]]
    x1_ = x1.groupby("min").last().reset_index()
#   x1_["min"] = pd.DataFrame(list(range(1,(x1_.shape[0]+1))),index=x1_.index)
    return x1_

def nf(x):
    x1 = be_ta[be_ta["Symbol"]==x.iloc[:,0].values[0]].loc[:,["Date","Industry","beta"]]
    xx = pd.merge(x, x1, left_on="date", right_on="Date", how="left").iloc[:,[4,1,5,6,2]]
    return xx

def cc(x):
    ii = in_dex[in_dex["StockID"]==x.iloc[:, 2].values[0]].loc[:,["date","min","time","returns"]]
    nn = pd.merge(ii, x, left_on="date", right_on="Date", how='left').iloc[:,[0,1,2,5,6,3,7,8]]
    nn["alpha"] = nn["s_returns"] - nn["beta"]*nn["returns"]
    return nn


# 1. transfer from tick to minute and save it in indexMinute.csv
#path = r'E:\indexTick'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = clean(pickle.load(F))
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_csv('E:\\indexMinute.csv', encoding="utf-8")

# 2. read stock minute date and save it into stockMinute.csv
#path = r'E:\IC'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F).iloc[:,[0,1,10,27]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_csv('E:\\stockMinute.csv', encoding="utf-8")

in_dex = pd.read_csv('E:\\indexMinute.csv', encoding="utf-8")
in_dex["returns"] = in_dex.groupby("StockID")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck = pd.read_csv('E:\\stockMinute.csv', encoding="utf-8")
sto_ck = sto_ck.drop(sto_ck.columns[0], axis=1)
sto_ck["returns"] = in_dex.groupby("StockID")['close'].apply(lambda x: x/x.shift(1)-1)
be_ta = pd.read_csv('E:\\beta.csv', encoding="utf-8").loc[:,["Symbol","Date","beta","Industry"]]
be_ta["beta"] = be_ta.groupby("Symbol")["beta"].apply(lambda x: x.shift(1))


test = pd.concat([sto_ck[sto_ck["StockID"]=="SH600000"],sto_ck[sto_ck["StockID"]=="SH600004"]])

xx = test.groupby("StockID")["StockID","min","returns","date"].apply(nf).reset_index().iloc[:,[2,3,0,4,5,6]]
#x = be_ta.groupby("Symbol")["Date","Symbol","Industry","beta","s_returns"].apply(cc)
#x = x.iloc[:,[2,3,0,5,9]]

print(xx)




