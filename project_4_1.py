import pandas as pd
import numpy as np
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

F1 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_CSI_60d.pkl",'rb')
F2 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_IF_60d.pkl",'rb')
F3 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_IC_60d.pkl",'rb')
csi_index_500 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_daily_SH000905.csv',
                            encoding = "utf-8").iloc[:,1:]
csi_index_300 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\index_daily_SH000300.csv',
                            encoding = "utf-8").iloc[:,1:]
csi_index_1000 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\index_daily_SH000852.csv',
                             encoding = "utf-8").iloc[:,1:]
csi_index_500_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv',
                        encoding = "utf-8")
csi_index_300_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000300.csv',
                            encoding = "utf-8")
csi_index_1000_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000852.csv',
                             encoding = "utf-8")

sto_ck = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]
CSI_beta = pickle.load(F1)
IF_beta = pickle.load(F2)
IC_beta = pickle.load(F3)
# 1. cut data frame to make date match
start_date = max(csi_index_300["Date"][0], csi_index_500["Date"][0], csi_index_1000["Date"][0],
                  CSI_beta.index[0], IF_beta.index[0], IC_beta.index[0], min(sto_ck["Date"]))
end_date = min(csi_index_300["Date"].iloc[-1], csi_index_500["Date"].iloc[-1], csi_index_1000["Date"].iloc[-1],
               CSI_beta.index[-1], IF_beta.index[-1], IC_beta.index[-1], max(sto_ck["Date"]))
#sto_ck = sto_ck[(sto_ck["Date"] <= end_date) & (sto_ck["Date"] >= start_date)]

# 2. calculate T-1 close ~ T open returns
sto_ck["returns"] = pd.Series(sto_ck.groupby("Symbol").apply(lambda x: (x['ret']/100 + 1)/(x['close']/x['open'])-1).values, index=sto_ck.index)
csi_index_500["returns"] = (csi_index_500['close']/csi_index_500['close'].shift(1))/(csi_index_500['close']/csi_index_500['open'])-1
csi_index_300["returns"] = (csi_index_300['close']/csi_index_300['close'].shift(1))/(csi_index_300['close']/csi_index_300['open'])-1
csi_index_1000["returns"] = (csi_index_1000['close']/csi_index_1000['close'].shift(1))/(csi_index_1000['close']/csi_index_1000['open'])-1
sto_ck = sto_ck.sort_values(by=["Date", "Symbol"])

# 3. calculate alpha each day
df = pd.DataFrame(columns=["Date", "ID", "index", "beta", "alphaLD1close_open", "stock_return", "index_return"])
date = sto_ck[(sto_ck["Date"] >= start_date) & (sto_ck["Date"] <= end_date)]["Date"].unique()
for i in range(1, len(date)):
    te_st = sto_ck[sto_ck["Date"] == date[i]]
    stocks = te_st["Symbol"]
    csi300 = pd.DataFrame(csi_index_300_com.columns[(csi_index_300_com[csi_index_300_com["Date"] == date[i]] != 0).values[0]][1:], columns=["Symbol"])
    csi500 = pd.DataFrame(csi_index_500_com.columns[(csi_index_500_com[csi_index_500_com["Date"] == date[i]] != 0).values[0]][1:], columns=["Symbol"])
    csiothers = pd.DataFrame(list(set(stocks) - set(csi300["Symbol"]) - set(csi500["Symbol"])), columns=["Symbol"])

    df1 = pd.merge(csi300, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
    beta1 = pd.DataFrame(IF_beta.loc[date[i - 1], :])
    beta1["Symbol"] = beta1.index
    beta1.columns = ['beta', 'Symbol']
    df1 = pd.merge(df1, beta1, left_on="Symbol", right_on="Symbol", how="inner")
    alpha = df1["returns"] - df1["beta"].multiply(
        csi_index_300[csi_index_300["Date"] == date[i]]["returns"].values[0], fill_value=1)
    df_1 = pd.DataFrame(
        {"Date": np.repeat(date[i], len(df1)), 'ID': df1["Symbol"].values, "index": np.repeat("SH000300", len(df1)),
         "beta": df1["beta"].values, "alphaLD1close_open": alpha.values, "stock_return": df1["returns"].values,
         "index_return": np.repeat(csi_index_300[csi_index_300["Date"] == date[i]]["returns"].values[0], len(df1))})

    df2 = pd.merge(csi500, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
    beta2 = pd.DataFrame(IC_beta.loc[date[i - 1], :])
    beta2["Symbol"] = beta2.index
    beta2.columns = ['beta', 'Symbol']
    df2 = pd.merge(df2, beta2, left_on="Symbol", right_on="Symbol", how="inner")
    alpha = df2["returns"] - df2["beta"].multiply(
        csi_index_500[csi_index_500["Date"] == date[i]]["returns"].values[0], fill_value=1)
    df_2 = pd.DataFrame(
        {"Date": np.repeat(date[i], len(df2)), 'ID': df2["Symbol"].values, "index": np.repeat("SH000905", len(df2)),
         "beta": df2["beta"].values, "alphaLD1close_open": alpha.values, "stock_return": df2["returns"].values,
         "index_return": np.repeat(csi_index_500[csi_index_500["Date"] == date[i]]["returns"].values[0], len(df2))
         })

    df3 = pd.merge(csiothers, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
    beta3 = pd.DataFrame(CSI_beta.loc[date[i - 1], :])
    beta3["Symbol"] = beta3.index
    beta3.columns = ['beta', 'Symbol']
    df3 = pd.merge(df3, beta3, left_on="Symbol", right_on="Symbol", how="inner")
    alpha = df3["returns"] - df3["beta"].multiply(
        csi_index_1000[csi_index_1000["Date"] == date[i]]["returns"].values[0], fill_value=1)
    df_3 = pd.DataFrame(
        {"Date": np.repeat(date[i], len(df3)), 'ID': df3["Symbol"].values, "index": np.repeat("SH000852", len(df3)),
         "beta": df3["beta"].values, "alphaLD1close_open": alpha.values, "stock_return": df3["returns"].values,
         "index_return": np.repeat(csi_index_1000[csi_index_1000["Date"] == date[i]]["returns"].values[0], len(df3))
         })

    df = pd.concat([df, df_1, df_2, df_3])
    print(df)

df = df.fillna(1.0)
df = df.sort_values(by=["Date", "ID"])
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table,'E:\\alpha_LD1close2open.parquet')

