import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm


def cal_cor(i, df, date):
    for i in range(1, len(date)-60):
        stocks1 = df[df["Date"] == date[i + 60]]
        stocks2 = df[df["Date"] == date[i + 59]]
        stocks = list(set(stocks1["ID"]) & set(stocks2["ID"]))
        test1 = df[(df["Date"] >= date[i]) & (df["Date"] < date[i + 60]) & (df["ID"].isin(stocks))]
        test2 = df[(df["Date"] >= date[i - 1]) & (df["Date"] < date[i + 59]) & (df["ID"].isin(stocks))]
        ndf1 = pd.DataFrame()
        ndf2 = pd.DataFrame()
        for j in stocks:
            if (len(test1[test1["ID"] == j]["alpha"]) == 60) & (len(test2[test2["ID"] == j]["alpha"]) == 60):
                ndf1[j] = pd.Series(test1[test1["ID"] == j]["alpha"].values, index=test1[test1["ID"] == j]["Date"])
                ndf2[j] = pd.Series(test2[test2["ID"] == j]["alpha"].values, index=test2[test2["ID"] == j]["Date"])

        df_1 = pd.DataFrame()
        for j in ndf1.columns:
            y = ndf1[j]
            ma_trix = pd.DataFrame()
            be_ta = []
            al_pha = []
            R = []
            li_st = []
            for j1 in ndf2.drop(j, axis=1).columns:
                X = pd.Series(ndf2[j1].values, index=y.index)
                X = sm.add_constant(X)
                est = sm.OLS(y, X)
                re = est.fit()
                be_ta.append(re.params.iloc[1])
                al_pha.append(re.params.iloc[0])
                R.append(re.rsquared)
                li_st.append(j1)
            ma_trix["Date"] = pd.Series(np.repeat(date[i + 59], len(li_st)))
            ma_trix["Target"] = pd.Series(np.repeat(j, len(li_st)))
            ma_trix["Paired_Stocks"] = pd.Series(li_st)
            ma_trix["beta"] = pd.Series(be_ta)
            ma_trix["alpha"] = pd.Series(al_pha)
            ma_trix["R2"] = pd.Series(R)
            # ma_trix["R^2"] = ma_trix["R^2"].rank(ascending=False)
            # ma_trix = ma_trix[ma_trix["R^2"] <= int(ndf1.shape[1] / 10)]
            # df_alpha = pd.concat([df_alpha, ma_trix["alpha"]], axis=1)
            # df_alpha = df_alpha.rename(columns={"alpha": j})
            # df_beta = pd.concat([df_beta, ma_trix["beta"]], axis=1)
            # df_beta = df_beta.rename(columns={"beta": j})
            df_1 = pd.concat([df_1, ma_trix])

        print(str(date[i+59]))
        df_1.to_csv("E:\\final result\\final_result_" + str(date[i+59]) + ".csv", encoding="utf-8")




df = pd.read_csv('E:\\daily_alpha.csv', encoding="utf-8").iloc[:, 1:]
date = df["Date"].unique()
df = df.sort_values(by=["Date", "ID"])
dff = pd.DataFrame(columns=["Date", "spread"])

cal_cor(1, df, date)

# for i in range(1, len(date) - 60, 5):
# #for i in range(1, 26, 5):
#     alpha, beta = cal_cor(i, df, date)
#     a1 = df[df["Date"] == date[i + 59]][df[df["Date"] == date[i + 59]]["ID"].isin(alpha.index)]["alpha"]
#     a2 = df[df["Date"] == date[i + 60]][df[df["Date"] == date[i + 60]]["ID"].isin(alpha.index)]["alpha"]
#     a1 = pd.DataFrame(a1.values, index=df[df["Date"] == date[i + 59]][df[df["Date"] ==
#                                                                          date[i + 59]]["ID"].isin(alpha.index)]["ID"])
#     prod = pd.concat([a1.mul(beta.iloc[:, i], axis=0) for i in range(beta.shape[1])], axis=1)
#     prod.set_axis(beta.columns, axis=1, inplace=True)
#     ret = prod + alpha
#     sell = ret.mean().sort_values().iloc[:int(len(ret)/5)].index.values
#     buy = ret.mean().sort_values().iloc[-int(len(ret)/5):].index.values
#     for j in range(i, i + 5):
#         spread = sum(df[df["Date"] == date[j + 60]][df[df["Date"] == date[j + 60]]["ID"].isin(buy)]["alpha"]) - \
#                  sum(df[df["Date"] == date[j + 60]][df[df["Date"] == date[j + 60]]["ID"].isin(sell)]["alpha"])
#         dff = dff.append({"Date": date[j + 60], "spread": spread}, ignore_index=True)
#
# print(dff)
# dff.to_csv("E:\\result.csv", encoding="utf-8")
#
# dff = pd.read_csv("E:\\result.csv", encoding="utf-8")
# print(dff.describe())
# dff["Date"] = dff["Date"].apply(lambda x: str(x)[:8])
# date_time = pd.to_datetime(dff["Date"])
# DF = pd.DataFrame()
# DF['spread'] = dff["spread"]
# DF = DF.set_index(date_time)
# DF1 = DF.resample('1M').mean()
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.3)
# plt.xticks(rotation=90)
# plt.plot(DF, 'b-', marker='.', label="daily alpha spread")
# plt.axhline(y=0, linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
# plt.plot(DF1, 'r-', label="monthly alpha spread")
# plt.title("Daily alpha result for strategy based on alpha forecast")
# plt.legend()
# plt.show()

