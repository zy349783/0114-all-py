import pandas as pd
import glob
import numpy as np

# def clean(x):
#     cl_ean = x["ListDays"] != 0
#     x = x[cl_ean]
#     return x.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
#
# path = r'E:\ST test'
# all_files = glob.glob(path + "/*.csv")
# dd = clean(pd.read_csv(all_files[0], encoding="GBK"))
# for i in range(1, len(all_files)):
#     dn = clean(pd.read_csv(all_files[i], encoding="GBK"))
#     dd = pd.concat([dd, dn], axis=0, ignore_index=True)
# dd = dd.sort_values(by=["Date", "Symbol"])
# dd.to_csv('E:\\all_stock_2019.csv', encoding="GBK")

ST = pd.read_csv("E:\\跌停统计1.csv", encoding="GBK")
ST.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
ST = ST.sort_values(by="首日")
da_te1 = ST["首日"].unique()
data = pd.read_csv('E:\\all_stock_2019.csv', encoding="GBK").iloc[:, 1:]
data["returns"] = data.groupby("Symbol")['close'].apply(lambda x: x/x.shift(1)-1)
drop = dict()
print(ST)
# create the basic portfolio
pf = pd.DataFrame()
da_te = data["Date"].unique()
hold_list = data[(data["Date"] == da_te[0]) & (data["trade_stats"] == 1)]["Symbol"]
de_list = data[(data["Date"] == da_te[0]) & (data["trade_stats"] == 0)]["Symbol"]
total_mv = data[(data["Date"] == da_te[0]) & (data["trade_stats"] == 1)]["MarketValue"].sum()
hold_money = (200000000/total_mv) * data[(data["Date"] == da_te[0]) & (data["trade_stats"] == 1)]["MarketValue"]
pf["ID"] = hold_list
pf["hold_value"] = hold_money
# deal with stocks add_in and drop_out
for i in range(1, len(da_te)):
    print(da_te[i])
    pf = pd.merge(pf, data[data["Date"] == da_te[i]].loc[:, ["Symbol", "returns", "MarketValue"]], left_on="ID",
                  right_on="Symbol", how="inner")
    pf["hold_value"] = pf["hold_value"] * (1 + pf["returns"])
    total_hold = pf["hold_value"].sum()
    hold_list = list(pf["ID"])
    hold_mv = list(pf["MarketValue"])
    # add_in
    list2 = data[(data["Date"] == da_te[i]) & (data["trade_stats"] == 1)]["Symbol"]
    if len(set(list2) - set(hold_list)) != 0:
        for j1 in list(set(list2) - set(hold_list)):
            hold_list.append(j1)
            hold_mv.append(data[(data["Date"] == da_te[i]) & (data["Symbol"] == j1)]["MarketValue"].values[0])
    # drop_out
    if len(drop.keys()) != 0:
        for j2 in drop.keys():
            if drop[j2] == i:
                del drop[j2]
                total_hold = total_hold - pf[pf["ID"] == j2]["hold_value"].values[0]
                hold_list.remove(j2)
                hold_mv.remove(pf[pf["ID"] == j2]["MarketValue"].values[0])
    if i in da_te1:
        for j3 in ST[ST["首日"] == i]["ID"].values:
            if ST[(ST["首日"] == i) & (ST["ID"] == j3)]["首日跌停"].values[0] == 0:
                total_hold = total_hold - pf[pf["ID"] == j3]["hold_value"].values[0]
                hold_list.remove(j3)
                hold_mv.remove(pf[pf["ID"] == j3]["MarketValue"].values[0])
            else:
                drop.update({j3: ST[(ST["首日"] == i) & (ST["ID"] == j3)]["末日"].values[0]})
    # rebalance
    hold_mv = np.asarray(hold_mv)
    total_mv = hold_mv.sum()
    hold_money = (total_hold / total_mv) * hold_mv
    del pf
    pf = pd.DataFrame()
    pf["ID"] = pd.Series(hold_list)
    pf["hold_value"] = pd.Series(hold_money)
    print(pf)

print("Benchmark Total P&L: " + str(pf["hold_value"].sum()))

# calculate the p&l if drop stocks 1, 5, 10 trading days earlier