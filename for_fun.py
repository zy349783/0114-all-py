import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy

sto_ck = pd.read_csv('E:\\beta.csv', encoding = "GBK").iloc[:, 1:]
test1 = sto_ck[sto_ck["Industry"] == "SW801780"]
test2 = copy.deepcopy(test1)
test2["s_returns"] = test1.groupby("Symbol")["s_returns"].shift(1)
stocks = test1["Symbol"].unique()
df = pd.DataFrame()
x1 = []
y1 = []
cor = []
coe = []
for i in stocks:
    for j in stocks:
        if i != j:
            data1 = pd.merge(test1[test1["Symbol"] == i], test2[test2["Symbol"] == j],
                             left_on="Date", right_on="Date", how="inner")
            data1 = data1.dropna(subset=["s_returns_x", "s_returns_y"])
            cor.append(data1.loc[:, ["s_returns_x", "s_returns_y"]].corr().iloc[0, 1])
            y = data1["s_returns_x"]
            X = data1["s_returns_y"]
            X = sm.add_constant(X)
            print(i)
            print(j)
            est = sm.OLS(y, X)
            est = est.fit()
            coe.append(est.params["s_returns_y"])
            print(coe)
            x1.append(i)
            y1.append(j)

df["stock_x"] = pd.Series(x1)
df["stock_y"] = pd.Series(y1)
df["correlation"] = pd.Series(cor)
df["coefficient"] = pd.Series(coe)
df = df.sort_values(by=["correlation"], ascending=False)
df = df.drop_duplicates(subset='correlation', keep='first')
df = df.reset_index().iloc[:, 1:]
df = df[df["correlation"] > 0.6]
# gp1 = df["stock_y"].unique()
df["returns"] = 0
for j in range(len(df)):
    da_te = list(set(test1.loc[test1["Symbol"] == df.loc[j, "stock_y"], "Date"]) &
                 set(test1.loc[test1["Symbol"] == df.loc[j, "stock_x"], "Date"]))
    da_te.sort()
    for i in range(1, len(da_te)):
        if test1[(test1["Symbol"] == df.loc[j, "stock_y"]) & (test1["Date"] == da_te[i-1])]["s_returns"].values[0] > 0:
            df.loc[j, "returns"] = df.loc[j, "returns"] + test1[(test1["Symbol"] == df.loc[j, "stock_x"]) &
                                                                (test1["Date"] == da_te[i])]["s_returns"].values[0]
        else:
            df.loc[j, "returns"] = df.loc[j, "returns"] - test1[(test1["Symbol"] == df.loc[j, "stock_x"]) &
                                                                 (test1["Date"] == da_te[i])]["s_returns"].values[0]
print(sum(df["returns"]))
