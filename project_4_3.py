import pandas as pd
import numpy as np
import pickle
import glob

def cal_factors(d, date):
    d["buy_gp"] = np.nan
    d["sale_gp"] = np.nan
    d.loc[(d["buy_amount"] < 50000) & (d["buy_amount"] > 0), "buy_gp"] = 1
    d.loc[(d["buy_amount"] >= 50000) & (d["buy_amount"] < 200000), "buy_gp"] = 2
    d.loc[(d["buy_amount"] >= 200000) & (d["buy_amount"] < 1000000), "buy_gp"] = 3
    d.loc[d["buy_amount"] >= 1000000, "buy_gp"] = 4
    d.loc[(d["sale_amount"] < 50000) & (d["sale_amount"] > 0), "sale_gp"] = 1
    d.loc[(d["sale_amount"] >= 50000) & (d["sale_amount"] < 200000), "sale_gp"] = 2
    d.loc[(d["sale_amount"] >= 200000) & (d["sale_amount"] < 1000000), "sale_gp"] = 3
    d.loc[d["sale_amount"] >= 1000000, "sale_gp"] = 4
    d1 = d.groupby(["StockID", "buy_gp"])["buy_amount"].sum().reset_index()
    d2 = d.groupby(["StockID", "sale_gp"])["sale_amount"].sum().reset_index()
    df1 = pd.DataFrame(columns=["StockID", "date", "buy_sm_amount", "sell_sm_amount", "buy_md_amount",
                                "sell_md_amount", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount",
                                "sell_elg_amount", "net_mf_amount"])
    df1["StockID"] = d1["StockID"].unique()
    df1["date"] = date
    df1["buy_sm_amount"] = pd.merge(df1, d1[d1["buy_gp"] == 1].loc[:, ["StockID", "buy_amount"]],
                                    left_on="StockID", right_on="StockID", how="left")["buy_amount"] / 10000
    df1["buy_md_amount"] = pd.merge(df1, d1[d1["buy_gp"] == 2].loc[:, ["StockID", "buy_amount"]],
                                    left_on="StockID", right_on="StockID", how="left")["buy_amount"] / 10000
    df1["buy_lg_amount"] = pd.merge(df1, d1[d1["buy_gp"] == 3].loc[:, ["StockID", "buy_amount"]],
                                    left_on="StockID", right_on="StockID", how="left")["buy_amount"] / 10000
    df1["buy_elg_amount"] = pd.merge(df1, d1[d1["buy_gp"] == 4].loc[:, ["StockID", "buy_amount"]],
                                     left_on="StockID", right_on="StockID", how="left")["buy_amount"] / 10000
    df1["sell_sm_amount"] = pd.merge(df1, d2[d2["sale_gp"] == 1].loc[:, ["StockID", "sale_amount"]],
                                     left_on="StockID", right_on="StockID", how="left")["sale_amount"] / 10000
    df1["sell_md_amount"] = pd.merge(df1, d2[d2["sale_gp"] == 2].loc[:, ["StockID", "sale_amount"]],
                                     left_on="StockID", right_on="StockID", how="left")["sale_amount"] / 10000
    df1["sell_lg_amount"] = pd.merge(df1, d2[d2["sale_gp"] == 3].loc[:, ["StockID", "sale_amount"]],
                                     left_on="StockID", right_on="StockID", how="left")["sale_amount"] / 10000
    df1["sell_elg_amount"] = pd.merge(df1, d2[d2["sale_gp"] == 4].loc[:, ["StockID", "sale_amount"]],
                                      left_on="StockID", right_on="StockID", how="left")["sale_amount"] / 10000
    df1["net_mf_amount"] = pd.merge(df1, (d.groupby("StockID")["sectional_buy_amount"].last()
                                          - d.groupby("StockID")["sectional_sale_amount"].last()).reset_index(),
                                    left_on="StockID", right_on="StockID", how="left").iloc[:, -1] / 10000
    return df1

path = r'E:\IC_TICK'
all_files = glob.glob(path + "/*.pkl")
df = pd.DataFrame()
for i in range(len(all_files)):
    F = open(all_files[i], 'rb')
    d = pickle.load(F)
    date = int(all_files[i][29:37])
    df1 = cal_factors(d, date)
    df = pd.concat([df, df1])
    print(df)
df.to_csv("E:\\factor_cal.csv", encoding="utf-8")



