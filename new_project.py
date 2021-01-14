import pandas as pd
import numpy as np
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

def clean(x):
    x["private_bv"] = x["buy_sm_vol"] + x["buy_md_vol"]
    x["private_sv"] = x["sell_sm_vol"] + x["sell_md_vol"]
    x["private_bm"] = x["buy_sm_amount"] + x["buy_md_amount"]
    x["private_sm"] = x["sell_sm_amount"] + x["sell_md_amount"]
    x["institution_bv"] = x["buy_lg_vol"] + x["buy_elg_vol"]
    x["institution_sv"] = x["sell_lg_vol"] + x["sell_elg_vol"]
    x["institution_bm"] = x["buy_lg_amount"] + x["buy_elg_amount"]
    x["institution_sm"] = x["sell_lg_amount"] + x["sell_elg_amount"]
    x["price"] = x["net_mf_amount"]/x["net_mf_vol"]*100
    x = x.drop(x.columns[[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]],axis = 1)
    return x

def statistics(x):
    price = x.iloc[:, 0]
    date = x.iloc[:, 1]
    mean = price.mean()
    std = price.std()
    num = np.arange(0, len(date))
    ft = date[date.apply(lambda x: str(x)[:4] == "2019")]
    if len(ft) != 0:
        n1 = num[date == ft.iloc[0]]
        row_num = x.shape[0]
        price1 = x.iloc[n1[0]:row_num, 0]
        mean_1y = price1.mean()
        std_1y = price1.std()
    else:
        mean_1y = np.nan
        std_1y = np.nan
    m_ax = max(price)
    m_in = min(price)
    dd = {'mean':mean,"std":std,"mean_1y":mean_1y,"std_1y":std_1y,"max":m_ax,"min":m_in}
    df = pd.DataFrame(dd, index=[0])
    return df

# 1. clean the stock data: date, symbol, industry, close; sort the stock; add returns
# save it into all_stock_daily.csv
#path = r'E:\data'
#all_files = glob.glob(path + "/*.csv")
#dd = pd.DataFrame()
#for i in range(len(all_files)):
    #dn = clean(pd.read_csv(all_files[i], encoding="utf-8"))
    #dd = dd.append(dn)
#dd = dd.sort_values(by=["StockID","date"])
#dd.to_csv('E:\\all_data.csv', encoding="utf-8")

# 2. create a dataframe to store basic stock statistical information
sto_ck = pd.read_csv('E:\\all_data.csv', encoding = "utf-8").iloc[:,1:]
re_turns = pd.read_csv('E:\\all_stock_daily.csv', encoding = "GBK").iloc[:,1:]
sto_ck = pd.merge(sto_ck, re_turns, left_on=["date","StockID"], right_on=["Date","Symbol"], how="inner")
sto_ck = sto_ck.drop(sto_ck.columns[[8, 13, 14]], axis=1)
sto_ck["returns"] = sto_ck.groupby("StockID")['close'].apply(lambda x: x/x.shift(1)-1)
info = sto_ck.groupby("StockID")["close","date"].apply(statistics).reset_index()
info = info.drop(info.columns[1], axis=1)
sto_ck = sto_ck.dropna(axis=0, how="any")

# 3. create several variables
# 1) 净流入额：总股数不变，净流入即价值增加 "net_mf_amount"
# 2）散户追涨杀跌行为：散户买入量-散户卖出量
sto_ck["zzsd"] = sto_ck["private_bv"] - sto_ck["private_sv"]
sto_ck["zzsd1"] = sto_ck["institution_bv"] - sto_ck["institution_sv"]
# 3）机构理性行为：机构卖出量/两者之和，机构买入量/两者之和
sto_ck["ins_buy_ratio"] = sto_ck["institution_bv"]/(sto_ck["institution_bv"]+sto_ck["private_bv"])
sto_ck["ins_sell_ratio"] = sto_ck["institution_sv"]/(sto_ck["institution_sv"]+sto_ck["private_sv"])
# 4）价格贵且流动性好的股票：机构买入额-机构卖出额
sto_ck["ins_net_m"] = sto_ck["institution_bm"] - sto_ck["institution_sm"]
# 5）流动性差的股票：机构买入额+个人买入额，机构卖出额+个人卖出额
sto_ck["total_bm"] = sto_ck["institution_bm"] + sto_ck["private_bm"]
sto_ck["total_sm"] = sto_ck["institution_sm"] + sto_ck["private_sm"]
# 6）随机：机构买入额+机构卖出额，在判断相反的情况下的市场反应
sto_ck["total_ins"] = sto_ck["institution_bm"] + sto_ck["institution_sm"]

# 4. do some research
def test(df):
    df1 = df.iloc[:,0]
    df2 = df.iloc[:,1]
    m1 = np.mean(df1)
    std1 = np.std(df1)
    m2 = np.mean(df2)
    std2 = np.std(df2)
    print(df.columns[0] + " And " + df.columns[1] + ":")
    print(df.columns[0] + '均值方差: mean=%.9f stdv=%.9f' % (m1, std1))
    print(df.columns[1] + '均值方差: mean=%.9f stdv=%.9f' % (m2, std2))
    sns.distplot(df1)
    plt.show()
    sns.distplot(df2)
    plt.show()
    plt.scatter(df1, df2)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.show()

    print(df.columns[0] +'正态性检验：', stats.kstest(df1, 'norm', (m1, std1)))
    print(df.columns[1] +'正态性检验：', stats.kstest(df2, 'norm', (m2, std2)))
    print('Pearson相关系数为：%.4f' % df.corr().iloc[0,1])
    lrModel = LinearRegression()
    model = lrModel.fit(df[[df.columns[0]]], df[[df.columns[1]]])
    print('coefficient of determination:', model.score(df[[df.columns[0]]], df[[df.columns[1]]]))
    print('alpha:', model.intercept_[0])
    print('beta:', model.coef_[0][0])
    data = pd.DataFrame([[df.columns[0], df.columns[1], m1, std1, m2, std2, df.corr().iloc[0,1], model.intercept_[0],
                          model.coef_[0][0], model.score(df[[df.columns[0]]], df[[df.columns[1]]])]], columns=["x_name",
                          "y_name", "x_mean", "x_std", "y_mean", "y_std", "corr", "alpha", "beta", "R^2"])
    return data

info = info.sort_values(["mean","std"], ascending=[False, False])
print(info)
# "SH600519":茅台；"SH601012":隆基股份；"SZ000860":顺鑫农业；"SH601155":新城控股；"SH601990":南京证券；"SZ300783":三只松鼠
sto_ck1 = sto_ck[sto_ck["StockID"] == "SZ300783"]
te_st = sto_ck1.loc[:,["returns","net_mf_amount","zzsd","zzsd1","ins_buy_ratio","ins_sell_ratio","ins_net_m","total_bm","total_sm","total_ins"]]
# 1) 净流入额：不能判断净流入量的变化，二者直接决定价格 "net_mf_amount"
d1 = te_st.loc[:,["returns","net_mf_amount"]]
# 2）散户追涨杀跌行为：散户买入量-散户卖出量
d2 = te_st.loc[:,["returns","zzsd"]]
d3 = te_st.loc[:,["returns","zzsd1"]]
# 3）机构理性行为：机构卖出量/两者之和，机构买入量/两者之和
d4 = te_st.loc[:,["returns","ins_buy_ratio"]]
d5 = te_st.loc[:,["returns","ins_sell_ratio"]]
# 4）价格贵且流动性好的股票：机构买入额-机构卖出额
d6 = te_st.loc[:,["returns","ins_net_m"]]
# 5）流动性差的股票：机构买入额+个人买入额，机构卖出额+个人卖出额
d7 = te_st.loc[:,["returns","total_bm"]]
d8 = te_st.loc[:,["returns","total_sm"]]
# 6）随机：机构买入额+机构卖出额，在判断相反的情况下的市场反应
d9 = te_st.loc[:,["returns","total_ins"]]
ss = pd.concat([test(d1), test(d2), test(d3), test(d4), test(d5), test(d6), test(d7), test(d8), test(d9)], axis = 0)
print(ss)


te_st["returns_lag1"] = te_st["returns"].shift(-1)
te_st = te_st.iloc[:(te_st.shape[0]-1),:]
d1 = te_st.loc[:,["returns_lag1","net_mf_amount"]]
d2 = te_st.loc[:,["returns_lag1","zzsd"]]
d3 = te_st.loc[:,["returns_lag1","zzsd1"]]
d4 = te_st.loc[:,["returns_lag1","ins_buy_ratio"]]
d5 = te_st.loc[:,["returns_lag1","ins_sell_ratio"]]
d6 = te_st.loc[:,["returns_lag1","ins_net_m"]]
d7 = te_st.loc[:,["returns_lag1","total_bm"]]
d8 = te_st.loc[:,["returns_lag1","total_sm"]]
d9 = te_st.loc[:,["returns_lag1","total_ins"]]
#print(pd.concat([test(d1), test(d2), test(d3), test(d4), test(d5), test(d6), test(d7), test(d8), test(d9)], axis = 0)["corr"])
