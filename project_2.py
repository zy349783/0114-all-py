import pandas as pd
import matplotlib as mpl
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def clean(x):
    x["private_bv"] = x["buy_sm_vol"] + x["buy_md_vol"]
    x["private_sv"] = x["sell_sm_vol"] + x["sell_md_vol"]
    x["private_bm"] = x["buy_sm_amount"] + x["buy_md_amount"]
    x["private_sm"] = x["sell_sm_amount"] + x["sell_md_amount"]
    x["institution_bv"] = x["buy_lg_vol"] + x["buy_elg_vol"]
    x["institution_sv"] = x["sell_lg_vol"] + x["sell_elg_vol"]
    x["institution_bm"] = x["buy_lg_amount"] + x["buy_elg_amount"]
    x["institution_sm"] = x["sell_lg_amount"] + x["sell_elg_amount"]
    x["total_buy_v"] = x["buy_sm_vol"] + x["buy_md_vol"] + x["buy_lg_vol"] + x["buy_elg_vol"]
    x["total_sell_v"] = x["sell_sm_vol"] + x["sell_md_vol"] + x["sell_lg_vol"] + x["sell_elg_vol"]
    x["total_buy_m"] = x["buy_sm_amount"] + x["buy_md_amount"] + x["buy_lg_amount"] + x["buy_elg_amount"]
    x["total_sell_m"] = x["sell_sm_amount"] + x["sell_md_amount"] + x["sell_lg_amount"] + x["sell_elg_amount"]
    x = x.drop(x.columns[[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]],axis = 1)
    return x

#path = r'E:\data'
#all_files = glob.glob(path + "/*.csv")
#dd = pd.DataFrame()
#for i in range(len(all_files)):
    #dn = clean(pd.read_csv(all_files[i], encoding="utf-8"))
    #dd = dd.append(dn)
#dd = dd.sort_values(by=["StockID","date"])
#dd.to_csv('E:\\all_data.csv', encoding="utf-8")

sto_ck = pd.read_csv('E:\\all_data.csv', encoding = "utf-8").iloc[:,1:]
re_turns = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]
in_dex = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1\\index_comp_SH000905.csv', encoding = "utf-8")
sto_ck = pd.merge(sto_ck, re_turns, left_on=["date","StockID"], right_on=["Date","Symbol"], how="inner")
sto_ck = sto_ck.drop(sto_ck.columns[[16, 17, 18]], axis=1)
sto_ck["returns"] = sto_ck.groupby("StockID")['close'].apply(lambda x: x/x.shift(1)-1)
sto_ck = sto_ck.dropna(axis=0, how="any")
che_ck = pd.read_csv('E:\\CheckVolumn.csv', encoding = "GBK").iloc[:,1:]
investors = pd.read_csv('E:\\investors.csv', encoding = "GBK").iloc[:,1:]
sto_ck = pd.merge(sto_ck, investors, left_on=["StockID"], right_on=["StockID"], how="inner")

# 1. Data cleaning and Sanity checks
# 1) Basic knowledge check
sto_ck["volume_diff"] = sto_ck["total_buy_v"] - sto_ck["total_sell_v"]
sto_ck["amount_diff"] = sto_ck["total_buy_m"] - sto_ck["total_sell_m"]
print(all(i == 0 for i in sto_ck["volume_diff"]))
print(all(i == 0 for i in sto_ck["amount_diff"]))
print("Distribution of volume difference:")
print(sto_ck["volume_diff"].describe())
print("Distribution of amount difference:")
print(sto_ck["amount_diff"].describe())
sns.distplot(sto_ck["volume_diff"])
plt.show()
sns.distplot(sto_ck["amount_diff"])
plt.show()
# 2) Database cross check
cc = pd.merge(sto_ck, che_ck, left_on=["date","StockID"], right_on=["Date","Symbol"], how="inner").loc[:,["StockID",
              "date","total_buy_m","total_buy_v","total_sell_m","total_sell_v","volume","amt"]]
cc["volume1"] = cc["total_buy_v"]*100
cc["amt1"] = cc["total_buy_m"]*10000
cc["vol_diff_ratio"] = (cc["volume"] - cc["volume1"])/cc["volume1"]
cc["amt_diff_ratio"] = (cc["amt"] - cc["amt1"])/cc["amt1"]
print(cc.iloc[:,[0,1,6,7,8,9]])
print("Distribution of volume difference ratio:")
print(cc["vol_diff_ratio"].describe())
print("Distribution of amount difference ratio:")
print(cc["amt_diff_ratio"].describe())
# There are Inf value inside cc, when the data from Tushare goes extra small
cc = cc.replace([np.inf, -np.inf], 0)
sns.distplot(cc["vol_diff_ratio"])
plt.show()
sns.distplot(cc["amt_diff_ratio"])
plt.show()

# 2. Descriptive Statistics
# 1) For a single stock: Distribution of the variables--大、特大买单/总买单
sto_ck = sto_ck[sto_ck["total_buy_m"]!=0]
sto_ck["ratio"] = sto_ck["institution_bm"]/sto_ck["total_buy_m"]
sto_ck1 = sto_ck[sto_ck["StockID"] == "SH600000"]
m1 = np.mean(sto_ck1["ratio"])
std1 = np.std(sto_ck1["ratio"])
print('大、特大买单占总买单比例的均值方差: mean=%.9f stdv=%.9f' % (m1, std1))
sns.distplot(sto_ck1["ratio"])
plt.show()
# 2) For a single stock: Stability across time
sto_ck1["date"] = sto_ck1["date"].apply(lambda x: str(x))
date_time = pd.to_datetime(sto_ck1["date"])
DF = pd.DataFrame()
DF['ratio'] = sto_ck1["ratio"]
DF = DF.set_index(date_time)
DF1 = DF.resample('1M').mean()
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)
plt.xticks(rotation=90)
plt.plot(DF, 'b-', label="daily ratio")
plt.plot(DF1, 'r-', label="monthly ratio")
plt.legend()
plt.show()
# 3) Distribution groupby Size or other stock characteristics
sto_ck["Percentile"] = sto_ck["MarketValue"].rank(pct=True)
bins = np.arange(0, 1.1, 0.1)
sto_ck["groups"] = pd.cut(sto_ck["Percentile"], bins)
df = sto_ck.groupby("groups")["ratio"].describe()
df.to_csv('E:\\lll.csv', encoding="utf-8")
# 4) Distribution groupby Individual Investor Percentage
stock_n = sto_ck[sto_ck["date"] == 20181228].loc[:,["ratio","ind_pc"]]
stock_n["Percentile"] = stock_n["ind_pc"].rank(pct=True)
bins = np.arange(0, 1.1, 0.1)
stock_n["groups"] = pd.cut(stock_n["Percentile"], bins)
df = stock_n.groupby("groups")["ratio"].describe()
df.to_csv('E:\\iii.csv', encoding="utf-8")

# 3. Relation with stock characteristics
# 1) With size-- simple regression
y = sto_ck["ratio"]
X = sto_ck["MarketValue"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
# 1) With size-- grouping
pp = sto_ck.groupby("groups")["ratio"].mean().reset_index()
pp["value"] = sto_ck.groupby("groups")["MarketValue"].mean().reset_index().iloc[:,1]
y = pp["ratio"]
X = pp["value"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
# 2) With individual investor percentage-- simple regression
y = stock_n["ratio"]
X = stock_n["ind_pc"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
# 2) With individual investor percentage-- grouping
pp = stock_n.groupby("groups")["ratio"].mean().reset_index()
pp["value"] = stock_n.groupby("groups")["ind_pc"].mean().reset_index().iloc[:,1]
y = pp["ratio"]
X = pp["value"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())

# 4. Explanatory power of current day return
# 1) For single stocks, time series relationship between factor and current day return
print("Explanatory power of current day return:")
print('Pearson相关系数为：%.4f' % sto_ck1.loc[:,["ratio","returns"]].corr().iloc[0,1])
y = sto_ck1["returns"]
X = sto_ck1["ratio"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
y_pre = est.predict(X)
plt.scatter(X.ratio, y, alpha=0.3)
plt.xlabel("ratio of institution buy over total buy")
plt.ylabel("current day returns")
plt.plot(X.ratio, y_pre, 'r', alpha=0.9)
plt.show()
# 2) Cross-sectional relation: Simple Regression
tt = in_dex[in_dex["Date"] == 20161230]
all_stocks = tt.columns[(tt!=0).all()][1:]
all_stocks = np.intersect1d(sto_ck["StockID"].unique(), all_stocks, assume_unique=True)
beta = pd.DataFrame()
dd = []
for i in all_stocks:
    stock = sto_ck[sto_ck["StockID"] == i]
    y = stock["returns"]
    X = stock["ratio"]
    X = sm.add_constant(X)
    est = sm.OLS(y, X)
    est = est.fit()
    dd.append(est.params[1])
beta["beta"] = pd.Series(dd, index=range(0,len(dd)))
beta["StockID"] = all_stocks
print(beta.describe())
sns.distplot(beta["beta"])
plt.show()
# 3) Cross-sectional relation: Grouping
data1 = pd.merge(beta, sto_ck, left_on="StockID", right_on="StockID", how="inner").groupby("StockID").last().reset_index().iloc[:,[0,1,17]]
data1["Percentile"] = data1["MarketValue"].rank(pct=True)
bins = np.arange(0, 1.1, 0.1)
data1["groups"] = pd.cut(data1["Percentile"], bins)
pp = data1.groupby("groups")["beta"].mean().reset_index()
pp["value"] = sto_ck.groupby("groups")["MarketValue"].mean().reset_index().iloc[:,1]
print(pp["beta"].describe())
sns.distplot(pp["beta"])
plt.show()
# 4) Extreme Case Analysis
ex_treme = sto_ck1[(sto_ck1["ratio"] > (m1+2*std1)) | (sto_ck1["ratio"] < (m1-2*std1))]
y = ex_treme["returns"]
X = ex_treme["ratio"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
y_pre = est.predict(X)
plt.scatter(X.ratio, y, alpha=0.3)
plt.xlabel("ratio of institution buy over total buy")
plt.ylabel("current day returns")
plt.plot(X.ratio, y_pre, 'r', alpha=0.9)
plt.show()

# 5. Forecasting power of ND1/ND5 return
#sto_ck1["returns_lag1"] = sto_ck1["returns"].shift(-1)
#te_st = sto_ck1.iloc[:(sto_ck1.shape[0]-1),:]
#y = te_st["returns_lag1"]
#X = te_st["ratio"]
#X = sm.add_constant(X)
#est = sm.OLS(y, X)
#est = est.fit()
#print(est.summary())
#y_pre = est.predict(X)
#plt.scatter(X.ratio, y, alpha=0.3)
#plt.xlabel("ratio of institution buy over total buy")
#plt.ylabel("ND1 returns")
#plt.plot(X.ratio, y_pre, 'r', alpha=0.9)
#plt.show()

sto_ck1["returns_lag5"] = sto_ck1["returns"].shift(-5)
te_st1 = sto_ck1.iloc[:(sto_ck1.shape[0]-5),:]
y = te_st1["returns_lag5"]
X = te_st1["ratio"]
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
print(est.summary())
y_pre = est.predict(X)
plt.scatter(X.ratio, y, alpha=0.3)
plt.xlabel("ratio of institution buy over total buy")
plt.ylabel("ND5 returns")
plt.plot(X.ratio, y_pre, 'r', alpha=0.9)
plt.show()