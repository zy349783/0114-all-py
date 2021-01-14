import numpy as np
import pandas as pd
import glob

def clean(x):
    cl_ean = x["ListDays"] != 0
    x = x[cl_ean]
    return x.iloc[:,[0,1,4,9]]

# 1. clean the stock data: date, symbol, industry, close; sort the stock; add returns
# save it into all_stock_daily.csv
#path = r'E:\stockDaily'
#all_files = glob.glob(path + "/*.csv")
#dd = clean(pd.read_csv(all_files[0], encoding = "GBK"))
#for i in range(1, len(all_files)):
    #dn = clean(pd.read_csv(all_files[i], encoding = "GBK"))
    #dd = pd.concat([dd, dn], axis = 0, ignore_index = True)
#dd = dd.sort_values(by=["Symbol","Date"])
#dd.to_csv('E:\\all_stock_daily.csv', encoding = "GBK")

# 2. clean the index data: date, returns
# save it into all_index_daily.csv
#path = r'E:\indexDaily'
#all_files = glob.glob(path + "/*.csv")
#id = pd.read_csv(all_files[0], encoding = "ISO-8859-1").iloc[2415:4804,[2,3]]
#id.columns = ["Date", all_files[0][26:34]]
#id.iloc[:,1] = id.iloc[:,1]/id.iloc[:,1].shift(1)-1
#for i in range(1, len(all_files)):
    #dn = pd.read_csv(all_files[i], encoding = "ISO-8859-1").iloc[2415:4804,3]
    #dn_= dn/dn.shift(1)-1
    #id[all_files[i][26:34]] = dn_
#id.to_csv('E:\\all_index_daily.csv', encoding = "ISO-8859-1")

in_dex = pd.read_csv('E:\\all_index_daily.csv', encoding = "ISO-8859-1").iloc[:,1:]
sto_ck = pd.read_csv('E:\\all_stock_daily.csv', encoding = "GBK").iloc[:,1:]

sto_ck.SWIndustry1 = sto_ck.SWIndustry1.map({"申万银行":"SW801780",
    "申万农林牧渔":"SW801010","申万采掘":"SW801020","申万化工":"SW801030",
    "申万电子":"SW801080","申万家用电器":"SW801110","申万食品饮料":"SW801120",
    "申万纺织服装":"SW801130","申万轻工制造":"SW801140", "申万医药生物":"SW801150",
    "申万公用事业":"SW801160","申万交通运输":"SW801170","申万房地产":"SW801180",
    "申万商业贸易":"SW801200","申万综合":"SW801230","申万有色金属":"SW801050",
    "申万钢铁":"SW801040","申万休闲服务":"SW801210","申万建筑材料":"SW801710",
    "申万电气设备":"SW801730","申万建筑装饰":"SW801720","申万机械设备":"SW801890",
    "申万国防军工":"SW801740","申万汽车":"SW801880","申万非银金融":"SW801790",
    "申万计算机":"SW801750","申万传媒":"SW801760","申万通信":"SW801770"})

s1 = sto_ck.copy()

def calc_beta(df):
    np_array = df.values
    s = np_array[:,1]
    m = np_array[:,2]
    covariance = np.cov(s,m)
    beta = covariance[0,1]/covariance[1,1]
    return beta

def rolling_apply(df, period, func):
    result = pd.Series(np.nan, index=df.index)
    if df.shape[0] > period:
        result[(period-1):df.shape[0]] = [func(df.iloc[(i-period+1):(i+1),:])
                                          for i in range((period-1), df.shape[0])]
    return result

def ff(x):
    x1 = x.iloc[:,[0,1]]
    x2 = in_dex[x.iloc[:,2].values[0]]
    x2_ = pd.concat([x2, in_dex["Date"]], axis=1)
    xx = x1.merge(x2_,left_on='Date',right_on='Date')
    xx.columns = ["Date", "x", "y"]
    if len(xx) == 0:
        re = [np.nan]
    else:
        re = rolling_apply(xx, 60, calc_beta)
        if len(re) < len(x1):
            re[len(re)] = np.nan
    return pd.DataFrame(re, index=range(0,len(re)))

def nf(x):
    x1 = x.iloc[:,[0,1]]
    x2 = in_dex[x.iloc[:,2].values[0]]
    x2_ = pd.concat([x2, in_dex["Date"]], axis=1)
    xx = x1.merge(x2_,left_on='Date',right_on='Date')
    xx.columns = ["Date","x","y"]
    if len(xx) == 0:
        re = [np.nan]
    else:
        xx_cov = xx.rolling(60).cov().unstack()['x']['y']
        xx_var = xx['y'].to_frame().rolling(60).var()
        result = xx_cov / xx_var.iloc[:, 0]
        if len(xx) < len(x1):
            result[len(result)] = np.nan
        re = result.values

    return pd.DataFrame(re, index=range(0,len(re)))

# solution one

sto_ck["returns"] = sto_ck.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck = sto_ck.dropna(axis=0, how="any")
sto_ck['beta'] = np.nan

#te_st = pd.concat([sto_ck[sto_ck['Symbol']=="SH600000"],sto_ck[sto_ck['Symbol']=="SH600004"],sto_ck[sto_ck['Symbol']=="SZ300795"],sto_ck[sto_ck['Symbol']=="SZ300797"]])
#beta1 = te_st.groupby("Symbol")['Date','returns','SWIndustry1'].apply(nf).reset_index().iloc[:,2].round(3)
#beta2 = te_st.groupby("Symbol")['Date','returns','SWIndustry1'].apply(ff).reset_index().iloc[:,2].round(3)
#print(beta1.equals(beta2))

#sto_ck["beta"]=sto_ck.groupby("Symbol")['Date','returns','SWIndustry1'].apply(nf).reset_index().iloc[:,2].round(3)
#print("beta result 1:")
#print(sto_ck)


# solution two
# connect two tables to prepare for beta calculation
def con(x):
    x2 = in_dex[x.iloc[:, 2].values[0]]
    x2_ = pd.concat([x2, in_dex["Date"]], axis=1)
    xx = pd.merge(x, x2_, how='left')
    xx.columns = ["Date", "stock", "Industry","i_returns"]
    return xx

s1 = s1.dropna(axis=0, how="any")
n_ew = s1.groupby("Symbol")['Date','close','SWIndustry1'].apply(con).reset_index().iloc[:,[0,2,3,4,5]]
n_ew = n_ew.sort_values(by=["Symbol","Date"])
n_ew["s_returns"] = n_ew.groupby("Symbol")['stock'].apply(lambda x: x/x.shift(1)-1)

# calculate beta
def cal(x):
    x.columns = ["x", "y"]
    x_cov = x.rolling(60, min_periods=30).cov().unstack()['x']['y']
    x_var = x['y'].to_frame().rolling(60, min_periods=30).var()
    result = x_cov / x_var.iloc[:, 0]
    re = pd.DataFrame(result.values, index=range(0, len(result)))
    print(re)
    return re
n_ew["beta"] = n_ew.groupby("Symbol")['s_returns','i_returns'].apply(cal).reset_index().iloc[:,2]
n_ew["beta"] = n_ew["beta"].shift(1)
n_ew["alpha"] = n_ew["s_returns"] - n_ew["beta"]*n_ew["i_returns"]
print("beta & alpha result 2:")
print(n_ew)

n_ew.to_csv('E:\\beta2.csv', encoding="utf-8")

