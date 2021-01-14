import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def lead_lag(frequency, data, tic1, tic2):
    if data.shape[0]==2389:
        data1 = data[tic1]
        data2 = data[tic2].shift(frequency)
        da_ta = pd.concat([data1, data2], axis=1)
        print(tic1 + ': mean=%.9f stdv=%.9f' % (np.mean(data1), np.std(data2)))
        print(tic2 + ': mean=%.9f stdv=%.9f' % (np.mean(data1), np.std(data2)))
        plt.scatter(data1, data2)
    else:
        data1 = data[data["StockID"] == tic1].loc[:, ["returns", "date", "time"]]
        data2 = data[data["StockID"] == tic2].loc[:, ["returns", "date", "time"]]
        data2.loc[:,"returns"] = data2["returns"].shift(frequency)
        da_ta = pd.merge(data1, data2, how="inner", left_on=["date", "time"], right_on=["date", "time"])
        print(tic1 + ': mean=%.9f stdv=%.9f' % (np.mean(da_ta.iloc[:, 0]), np.std(da_ta.iloc[:, 0])))
        print(tic2 + ': mean=%.9f stdv=%.9f' % (np.mean(da_ta.iloc[:, 3]), np.std(da_ta.iloc[:, 3])))
        plt.scatter(da_ta.iloc[:, 0], da_ta.iloc[:, 3])

    # basic statistics
    plt.xlabel(tic1)
    plt.ylabel(tic2)
    plt.show()
    # calculate covariance
    cov = da_ta.cov()
    print('Covariance: %.9f' % cov.iloc[0,1])
    # calculate correlation (strength of linear relationship)
    corr1 = da_ta.corr()
    print(tic1+' & '+tic2+'.lag%x' % frequency + ' ' + 'Correlation: %.9f' % corr1.iloc[0,1])
    return 0

def lead_lag_any(frequency, data, tic1, tic2):
    if data.shape[0] == 2389:
        data1 = data[tic1]
        data2 = data[tic2].shift(frequency)
        da_ta = pd.concat([data1, data2], axis=1)
    else:
        data1 = data[data["StockID"] == tic1].loc[:, ["returns", "date", "time"]]
        data2 = data[data["StockID"] == tic2].loc[:, ["returns", "date", "time"]]
        data2.loc[:,"returns"] = data2["returns"].shift(frequency)
        da_ta = pd.merge(data1, data2, how="inner", left_on=["date", "time"], right_on=["date", "time"])
    corr1 = da_ta.corr()
    return pd.DataFrame([[tic1, tic2, frequency, corr1.iloc[0, 1]]], columns=["tic1", "tic2", "lag", "correlation"])

d_index = pd.read_csv('E:\\all_index_daily.csv', encoding="utf-8")
m_index = pd.read_csv('E:\\indexMinute.csv', encoding="utf-8")
m_index["returns"] = m_index.groupby("StockID")['close'].apply(lambda x: x/x.shift(1)-1)
all_industries = d_index.columns[2:]
all_cob = list(itertools.combinations(all_industries, 2))

# 1. "SW801120" & "SW801750" industry lead-lag relationship
# lead_lag(1, d_index, "SW801120", "SW801750")
# lead_lag(3, d_index, "SW801120", "SW801750")
# lead_lag(5, d_index, "SW801120", "SW801750")
# lead_lag(1, m_index, "SW801120", "SW801750")
# lead_lag(3, m_index, "SW801120", "SW801750")
# lead_lag(5, m_index, "SW801120", "SW801750")

# 2. cross industry lead-lag relationship
x1 = lead_lag_any(1, d_index, all_cob[0][0], all_cob[0][1])
x2 = lead_lag_any(1, d_index, all_cob[0][1], all_cob[0][0])
x = pd.concat([x1, x2], axis=0)

for lag in [3, 5]:
    x1 = lead_lag_any(lag, d_index, all_cob[0][0], all_cob[0][1])
    x2 = lead_lag_any(lag, d_index, all_cob[0][1], all_cob[0][0])
    x = pd.concat([x, x1, x2], axis=0)

for i in all_cob[1:len(all_cob)]:
    for lag in [1, 3, 5]:
        x1 = lead_lag_any(lag, d_index, i[0], i[1])
        x2 = lead_lag_any(lag, d_index, i[1], i[0])
        x = pd.concat([x, x1, x2], axis=0)

x.tic1 = x.tic1.map({"SW801780":"申万银行", "SW801010":"申万农林牧渔",
    "SW801020":"申万采掘","SW801030":"申万化工","SW801080":"申万电子",
    "SW801110":"申万家用电器","SW801120":"申万食品饮料","SW801130":"申万纺织服装",
    "SW801140":"申万轻工制造", "SW801150":"申万医药生物", "SW801160":"申万公用事业",
    "SW801170":"申万交通运输", "SW801180":"申万房地产", "SW801230":"申万综合",
    "SW801200":"申万商业贸易", "SW801050":"申万有色金属","SW801040":"申万钢铁",
    "SW801210":"申万休闲服务", "SW801710":"申万建筑材料", "SW801730":"申万电气设备",
    "SW801720":"申万建筑装饰", "SW801890":"申万机械设备", "SW801740":"申万国防军工",
    "SW801880":"申万汽车", "SW801790":"申万非银金融", "SW801750":"申万计算机",
    "SW801760":"申万传媒", "SW801770":"申万通信"})
x.tic2 = x.tic2.map({"SW801780":"申万银行", "SW801010":"申万农林牧渔",
    "SW801020":"申万采掘","SW801030":"申万化工","SW801080":"申万电子",
    "SW801110":"申万家用电器","SW801120":"申万食品饮料","SW801130":"申万纺织服装",
    "SW801140":"申万轻工制造", "SW801150":"申万医药生物", "SW801160":"申万公用事业",
    "SW801170":"申万交通运输", "SW801180":"申万房地产", "SW801230":"申万综合",
    "SW801200":"申万商业贸易", "SW801050":"申万有色金属","SW801040":"申万钢铁",
    "SW801210":"申万休闲服务", "SW801710":"申万建筑材料", "SW801730":"申万电气设备",
    "SW801720":"申万建筑装饰", "SW801890":"申万机械设备", "SW801740":"申万国防军工",
    "SW801880":"申万汽车", "SW801790":"申万非银金融", "SW801750":"申万计算机",
    "SW801760":"申万传媒", "SW801770":"申万通信"})
x = x.sort_values(by='correlation', ascending=False)
print(x[x["correlation"] > 0.8])
print(x)


