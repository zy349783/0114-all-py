import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt

# F1 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_IC_60d.pkl", 'rb')
# sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
# csi_index_500 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_daily_SH000905.csv',
#                             encoding="utf-8").iloc[:,1:]
# csi_index_500_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv',
#                         encoding="utf-8")
# IC_beta = pickle.load(F1)
# start_date = max(csi_index_500["Date"][0], IC_beta.index[0], min(sto_ck["Date"]))
# end_date = min(csi_index_500["Date"].iloc[-1], IC_beta.index[-1], max(sto_ck["Date"]))
# csi_index_500["returns"] = csi_index_500['close']/csi_index_500['close'].shift(1) - 1
# sto_ck["ret"] = sto_ck['ret']/100
# sto_ck = sto_ck.sort_values(by=["Date", "Symbol"])
# df = pd.DataFrame(columns=["Date", "ID", "beta", "alpha", "stock_return", "index_return"])
# date = sto_ck[(sto_ck["Date"] >= 20181001) & (sto_ck["Date"] <= end_date)]["Date"].unique()
# for i in range(1, len(date)):
#     te_st = sto_ck[sto_ck["Date"] == date[i]]
#     stocks = te_st["Symbol"]
#     csi500 = pd.DataFrame(
#     csi_index_500_com.columns[(csi_index_500_com[csi_index_500_com["Date"] == date[i]] != 0).values[0]][1:],
#         columns=["Symbol"])
#     df2 = pd.merge(csi500, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "ret"]]
#     beta2 = pd.DataFrame(IC_beta.loc[date[i - 1], :])
#     beta2["Symbol"] = beta2.index
#     beta2.columns = ['beta', 'Symbol']
#     df2 = pd.merge(df2, beta2, left_on="Symbol", right_on="Symbol", how="inner")
#     alpha = df2["ret"] - df2["beta"].multiply(
#         csi_index_500[csi_index_500["Date"] == date[i]]["returns"].values[0], fill_value=1)
#     df_2 = pd.DataFrame(
#         {"Date": np.repeat(date[i], len(df2)), 'ID': df2["Symbol"].values, "beta": df2["beta"].values,
#          "alpha": alpha.values, "stock_return": df2["ret"].values,
#          "index_return": np.repeat(csi_index_500[csi_index_500["Date"] == date[i]]["returns"].values[0], len(df2))
#          })
#     df = pd.concat([df, df_2])
#     print(df)
#
# df.to_csv("E:\\daily_alpha.csv", encoding="utf-8")

# df = pd.read_csv('E:\\daily_alpha.csv', encoding="utf-8").iloc[:, 1:]
# date = df["Date"].unique()
# df = df.sort_values(by=["Date", "ID"])
# dff = pd.DataFrame()
# for i in range(1, len(date) - 60):
#     stocks1 = df[df["Date"] == date[i + 60]]
#     stocks2 = df[df["Date"] == date[i + 59]]
#     stocks = list(set(stocks1["ID"]) & set(stocks2["ID"]))
#     test1 = df[(df["Date"] >= date[i]) & (df["Date"] < date[i + 60]) & (df["ID"].isin(stocks))]
#     test2 = df[(df["Date"] >= date[i - 1]) & (df["Date"] < date[i + 59]) & (df["ID"].isin(stocks))]
#     ndf1 = pd.DataFrame()
#     ndf2 = pd.DataFrame()
#     for j in stocks:
#         if (len(test1[test1["ID"] == j]["alpha"]) == 60) & (len(test2[test2["ID"] == j]["alpha"]) == 60):
#             ndf1[j] = pd.Series(test1[test1["ID"] == j]["alpha"].values, index=test1[test1["ID"] == j]["Date"])
#             ndf2[j] = pd.Series(test2[test2["ID"] == j]["alpha"].values, index=test2[test2["ID"] == j]["Date"])
#     cor_matrix = pd.DataFrame()
#     for j in ndf1.columns:
#         cor_matrix[j] = pd.concat([ndf1[j], pd.DataFrame(ndf2.drop(j, axis=1).values, index=ndf1.index,
#                         columns=ndf2.drop(j, axis=1).columns)], axis=1).corr().iloc[:, 0]
#     # cor_matrix = ndf1.apply(lambda x: pd.DataFrame(ndf2.values, index=ndf1.index).corrwith(x))
#     # cor_matrix.index = cor_matrix.columns
#     np.fill_diagonal(cor_matrix.values, np.nan)
#     cor_matrix = cor_matrix.apply(lambda x: pd.cut(x.rank(pct=True), np.arange(0, 1.1, 0.1)))
#     cor_matrix["ID"] = cor_matrix.index
#     cor_matrix = pd.merge(cor_matrix, stocks1.loc[:, ["ID", "alpha"]], left_on=["ID"],
#                           right_on=["ID"], how="left")
#     cor_matrix = pd.merge(cor_matrix, stocks2.loc[:, ["ID", "alpha"]], left_on=["ID"],
#                           right_on=["ID"], how="left")
#     dff1 = cor_matrix.iloc[:, :len(cor_matrix)].apply(lambda x: cor_matrix.groupby(x)["alpha_y"].mean()).reset_index().T
#     dff1.columns = ["gp1", "gp2", "gp3", "gp4", "gp5", "gp6", "gp7", "gp8", "gp9", "gp10"]
#     dff1 = dff1.iloc[1:, :]
#     dff1["alpha"] = pd.Series(cor_matrix["alpha_x"].values, index=dff1.index)
#     dff1["Date"] = date[i + 60]
#     dff = dff.append(dff1)
#     print(dff)
#
# dff.to_csv("E:\\group_return.csv", encoding="utf-8")

all_alpha = pd.read_csv('E:\\daily_alpha.csv', encoding="GBK").iloc[:, 1:]
dff = pd.read_csv('E:\\group_return.csv', encoding="GBK")
dff.rename(columns={'Unnamed: 0': 'Target'}, inplace=True )
dff2 = pd.DataFrame(columns=["Target", "gp1", "gp2", "gp3", "gp4", "gp5", "gp6", "gp7", "gp8", "gp9", "gp10"])
for i in dff["Target"].unique():
    cor1 = dff[dff["Target"] == i].loc[:, ["alpha", "gp1"]].corr().iloc[0, 1]
    cor2 = dff[dff["Target"] == i].loc[:, ["alpha", "gp2"]].corr().iloc[0, 1]
    cor3 = dff[dff["Target"] == i].loc[:, ["alpha", "gp3"]].corr().iloc[0, 1]
    cor4 = dff[dff["Target"] == i].loc[:, ["alpha", "gp4"]].corr().iloc[0, 1]
    cor5 = dff[dff["Target"] == i].loc[:, ["alpha", "gp5"]].corr().iloc[0, 1]
    cor6 = dff[dff["Target"] == i].loc[:, ["alpha", "gp6"]].corr().iloc[0, 1]
    cor7 = dff[dff["Target"] == i].loc[:, ["alpha", "gp7"]].corr().iloc[0, 1]
    cor8 = dff[dff["Target"] == i].loc[:, ["alpha", "gp8"]].corr().iloc[0, 1]
    cor9 = dff[dff["Target"] == i].loc[:, ["alpha", "gp9"]].corr().iloc[0, 1]
    cor10 = dff[dff["Target"] == i].loc[:, ["alpha", "gp10"]].corr().iloc[0, 1]
    dff2 = dff2.append({"Target": i, "gp1": cor1, "gp2": cor2, "gp3": cor3, "gp4": cor4, "gp5": cor5, "gp6": cor6,
                        "gp7": cor7, "gp8": cor8, "gp9": cor9, "gp10": cor10}, ignore_index=True)

cols = ["Target", "gp1", "gp2", "gp3", "gp4", "gp5", "gp6", "gp7", "gp8", "gp9", "gp10"]
dff22 = dff2[cols]
plt.bar(np.arange(10), dff22.iloc[0, 1:].abs().values)
plt.xticks(np.arange(10), cols[1:])
plt.title("The correlation of " + dff22.iloc[0, 0] + " return and all other groups of stocks' mean return")
plt.show()
plt.bar(np.arange(10), dff22.iloc[:, 1:].abs().mean().values)
plt.xticks(np.arange(10), cols[1:])
plt.title("The correlation of stock return and all other groups of stocks' mean return")
plt.show()
print(dff22)
