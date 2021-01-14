import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from numpy import linalg
from matplotlib import pyplot as plt
from matplotlib.ticker import Formatter

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

F1 = open("E:\\公募基金净值数据.pkl", 'rb')
F2 = open("C:\\Users\\win\\Downloads\\Fundholding.pkl", 'rb')
F3 = open("C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\指数.pkl", 'rb')
df1 = pickle.load(F1)
df2 = pickle.load(F2)
df3 = pickle.load(F3)
ty_pe = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 6 基金持仓预测\\fund_type.csv', encoding="GBK")
ty_pe["证券代码"] = ty_pe["证券代码"].apply(lambda x: x.split('.')[1] + x.split('.')[0])
df1["ts_code"] = df1["ts_code"].apply(lambda x: x.split('.')[1] + x.split('.')[0])

# 1. 筛选出股票型基金作为测算对象, 728只公募基金
# li_st = ty_pe[(ty_pe["证监会基金类型"] == "股票型基金") & (ty_pe["证券代码"].str[:2] != "OF")]["证券代码"].values
li_st = ty_pe[ty_pe["证监会基金类型"] == "股票型基金"]["证券代码"].values
li_st1 = df1[df1["ts_code"].isin(li_st)]["ts_code"].unique()
df2 = df2[df2["FundID"].isin(li_st1)]
li_st = df2["FundID"].unique()
df1 = df1[df1["ts_code"].isin(li_st)]

# 2. 样本内测试（样本内：2015.09.30 - 2018.12.31）
# 2.1 PCA处理多重共线性
df1["end_date"] = df1["end_date"].astype(int)
# df1 = df1[df1["end_date"] >= 20150930]
df1 = df1.sort_values(by=["ts_code", "end_date"])
df1["returns"] = df1.groupby("ts_code")['unit_nav'].apply(lambda x: x/x.shift(1)-1)
df3["交易时间"] = df3["交易时间"].apply(lambda x: int((x.replace('-', "")).replace("-", "")))
df3 = df3.loc[:, ["证券代码", "交易时间", "收盘价"]]
df3["收盘价"] = df3["收盘价"].apply(lambda x: float(x.replace(',', "")))
df3["收益率"] = df3.groupby("证券代码")['收盘价'].apply(lambda x: x/x.shift(1)-1)

def fn(A, X, y):
    return np.linalg.norm(X.dot(A) - y)

def fn1(A, X, y, lam):
    return (np.linalg.norm(X.dot(A) - y)) ** 2 + lam * ((np.linalg.norm(A)) ** 2)

def fn2(A, X, y, lam):
    return ((X.dot(A) - y)**2).sum() + lam * (A**2).sum()

def fn3(A, X, y, lam):
    return ((X.dot(A) - y)**2).sum() + lam * np.linalg.norm(A, ord=1)

def PCA_unrestricted(N, M):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - M + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - M + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = StandardScaler().fit_transform(X)
        pca = PCA(26)
        X = pca.fit_transform(X)
        X = sm.add_constant(X)
        X = X[len(X) - N:, :]
        y = y.iloc[len(y) - N:, :]
        for i in range(y.shape[1]):
            est = sm.OLS(y.iloc[:, i], X)
            est = est.fit()
            cw.append(est.params.sum() - est.params["const"])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

def PCA_restricted(N, M):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - M + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - M + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = StandardScaler().fit_transform(X)
        pca = PCA(26)
        X = pca.fit_transform(X)
        X = sm.add_constant(X)
        X = X[len(X) - N:, :]
        y = y.iloc[len(y) - N:, :]
        for i in range(y.shape[1]):
            yy = y.iloc[:, i].values
            x0, rnorm = nnls(X, yy)
            cons = {'type': 'ineq', 'fun': lambda x: 0.95 - np.sum(x) + x[0]}
            bounds = [[0.05, 1], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None]]
            minout = minimize(fn, x0, args=(X, yy), method='SLSQP', bounds=bounds, constraints=cons)
            cw.append(minout.x.sum() - minout.x[0])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

def Ridge_unrestricted(N, A):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - N + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - N + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = sm.add_constant(X)
        for i in range(y.shape[1]):
            rr = Ridge(alpha=A, normalize=True)
            rr = rr.fit(X, y.iloc[:, i])
            cw.append(rr.coef_.sum()-rr.coef_[0])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

def Ridge_restricted(N, A):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - N + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - N + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = sm.add_constant(X)
        for i in range(y.shape[1]):
            rr = Ridge(alpha=A, normalize=True)
            rr = rr.fit(X, y.iloc[:, i])
            x0 = rr.coef_
            cons = {'type': 'ineq', 'fun': lambda x: 0.95 - np.sum(x) + x[0]}
            bounds = [[0, None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None]]
            minout = minimize(fn2, x0, args=(X, y.iloc[:, i], A), method='SLSQP', bounds=bounds, constraints=cons)
            cw.append(minout.x.sum() - minout.x[0])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

def Lasso_unrestricted(N, A):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - N + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - N + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = sm.add_constant(X)
        for i in range(y.shape[1]):
            rr = Lasso(alpha=A, normalize=True)
            rr = rr.fit(X, y.iloc[:, i])
            cw.append(rr.coef_.sum()-rr.coef_[0])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

def Lasso_restricted(N, A):
    date_list = np.sort(df2[(df2["TargetDate"] > 20150930) & (df2["TargetDate"] < 20180930)]["TargetDate"].unique())[::2]
    diff = []
    date = df3["交易时间"].unique()
    for t1 in date_list:
        cw = []
        if t1 in date:
            pos = np.where(date == t1)[0][0] - N + 1
            t3 = t1
        else:
            t3 = t1
            t1 = date[np.where(date <= t1)[0][-1]]
            pos = np.where(date == t1)[0][0] - N + 1
        t2 = date[pos]
        X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
        X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
        y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
        y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
        X = X.iloc[:, 1:]
        X = sm.add_constant(X)
        for i in range(y.shape[1]):
            rr = Ridge(alpha=A, normalize=True)
            rr = rr.fit(X, y.iloc[:, i])
            x0 = rr.coef_
            cons = {'type': 'ineq', 'fun': lambda x: 0.95 - np.sum(x) + x[0]}
            bounds = [[0, None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
                      [0., None], [0., None], [0., None], [0., None], [0., None], [0., None]]
            minout = minimize(fn2, x0, args=(X, y.iloc[:, i], A), method='SLSQP', bounds=bounds, constraints=cons)
            cw.append(minout.x.sum() - minout.x[0])
        re = pd.DataFrame()
        re["fund"] = y.columns
        re["cw"] = cw
        real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
            "ratio"].sum().reset_index()
        re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
        diff.append(re["cw"].mean() * 100 - re["ratio"].mean())
    diff = [abs(x) for x in diff]
    return np.mean(diff)

# print(PCA_restricted(20, 125))
# print(Ridge_unrestricted(50, 35))
# print(Ridge_restricted(50, 35))

# print(Lasso_unrestricted(29, 0.0001))
# print(Lasso_restricted(29, 0.0001))
# print(Lasso_restricted(90, 0.0001))
# print(Lasso_restricted(29, 0.001)) 当前最好
# print(Lasso_restricted(90, 0.001))
# print(Lasso_restricted(29, 0.01))
# print(Lasso_restricted(90, 0.01))
pp = pd.DataFrame()
for i in range(30, 100, 10):
    L = []
    for j in [0.0008, 0.0009, 0.001, 0.002, 0.003]:
        re = Ridge_restricted(i, j)
        L.append(re)
    pp[str(i)] = L
pp.to_csv("E:\\biubiu.csv", encoding="utf-8")


# plot the best condition
# df = pd.DataFrame()
# N = 90
# A = 0.0007
# date_list = np.sort(df2[(df2["TargetDate"] > 20150930)]["TargetDate"].unique())[::2]
# r1 = []
# r2 = []
# date = df3["交易时间"].unique()
# for t1 in date_list:
#     cw = []
#     if t1 in date:
#         pos = np.where(date == t1)[0][0] - N + 1
#         t3 = t1
#     else:
#         t3 = t1
#         t1 = date[np.where(date <= t1)[0][-1]]
#         pos = np.where(date == t1)[0][0] - N + 1
#     t2 = date[pos]
#     X = pd.pivot_table(df3, values='收益率', index='交易时间', columns='证券代码').reset_index()
#     X = X[(X["交易时间"] >= t2) & (X["交易时间"] <= t1)]
#     y = pd.pivot_table(df1, values="returns", index="end_date", columns='ts_code').reset_index()
#     y = pd.merge(X["交易时间"], y, left_on="交易时间", right_on="end_date", how="left").dropna(axis="columns").iloc[:, 2:]
#     X = X.iloc[:, 1:]
#     X = sm.add_constant(X)
#     for i in range(y.shape[1]):
#         rr = Ridge(alpha=A, normalize=True)
#         rr = rr.fit(X, y.iloc[:, i])
#         x0 = rr.coef_
#         cons = {'type': 'ineq', 'fun': lambda x: 0.95 - np.sum(x) + x[0]}
#         bounds = [[0, None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
#                   [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
#                   [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None], [0., None],
#                   [0., None], [0., None], [0., None], [0., None], [0., None], [0., None]]
#         minout = minimize(fn2, x0, args=(X, y.iloc[:, i], A), method='SLSQP', bounds=bounds, constraints=cons)
#         cw.append(minout.x.sum() - minout.x[0])
#     re = pd.DataFrame()
#     re["fund"] = y.columns
#     re["cw"] = cw
#     real = df2[(df2["FundID"].isin(y.columns)) & (df2["TargetDate"] == t3)].groupby("FundID")[
#         "ratio"].sum().reset_index()
#     re = pd.merge(re, real, left_on=["fund"], right_on=["FundID"], how="inner")
#     r1.append(re["cw"].mean() * 100)
#     r2.append(re["ratio"].mean())
# df["Date"] = date_list
# df["cw"] = r1
# df["real"] = r2
# df["diff"] = (df["cw"] - df["real"]).abs()
# class MyFormatter(Formatter):
#     def __init__(self, dates, fmt='%Y%m'):
#         self.dates = dates
#         self.fmt = fmt
#
#     def __call__(self, x, pos=0):
#         """Return the label for time x at position pos"""
#         ind = int(np.round(x))
#         if ind >= len(self.dates) or ind < 0:
#             return ''
#         return pd.to_datetime(self.dates[ind], format="%Y%m%d").strftime(self.fmt)
#
# fig, ax = plt.subplots(figsize=(15, 6))
# ax.plot(np.arange(len(df)), df["cw"], color='blue', alpha=4, linewidth=1, linestyle='-', marker='.',
#         markersize=4, label='测算值')
# ax.plot(np.arange(len(df)), df["real"], color='red', alpha=4, linewidth=1, linestyle='-', marker='.',
#         markersize=4, label='真实值')
# ax.set_xlabel('')
# ax.set_ylabel('%')
# ax.legend(loc='upper right')
# ax.xaxis.set_major_formatter(MyFormatter(df["Date"].values, '%Y%m%d'))
# ax.set_title("股票型基金仓位Lasso回归测算效果（N=90， A=0.01）")
# ax.grid()
# plt.show()




print("llll")
print(df1)