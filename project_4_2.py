import pandas as pd
import numpy as np
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import glob

# 1. read stock minute date
#F = open("E:\\index\\INDEX_MIN_TICK_SH000300.pkl",'rb')
#d = pickle.load(F)
#d = d.loc[:, ['date', 'min', 'close']]
#d.to_pickle("E:\\INDEX_MIN_TICK_SH000300.pkl")
#F = open("E:\\index\\INDEX_MIN_TICK_SH000852.pkl",'rb')
#d = pickle.load(F)
#d = d.loc[:, ['date', 'min', 'close']]
#d.to_pickle("E:\\INDEX_MIN_TICK_SH000852.pkl")
#F = open("E:\\index\\INDEX_MIN_TICK_SH000905.pkl",'rb')
#d = pickle.load(F)
#d = d.loc[:, ['date', 'min', 'close']]
#d.to_pickle("E:\\INDEX_MIN_TICK_SH000905.pkl")

#path = r'E:\stock\CSI'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F)
    #dn = dn.loc[:, ["StockID", "date", "min", "buy1", "sale1", "open", "close"]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_pickle('E:\\stock_minute_CSI.pkl')

#path = r'E:\stock\CSIRest'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F)
    #dn = dn.loc[:, ["StockID", "date", "min", "buy1", "sale1", "open", "close"]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_pickle('E:\\stock_minute_CSIRest.pkl')

#path = r'E:\stock\IC'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F)
    #dn = dn.loc[:, ["StockID", "date", "min", "buy1", "sale1", "open", "close"]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_pickle('E:\\stock_minute_IC.pkl')

#path = r'E:\stock\IF'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F)
    #dn = dn.loc[:, ["StockID", "date", "min", "buy1", "sale1", "open", "close"]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_pickle('E:\\stock_minute_IF.pkl')

#path = r'E:\stock\Rest'
#all_files = glob.glob(path + "/*.pkl")
#d1 = pd.DataFrame()
#for i in range(len(all_files)):
    #F = open(all_files[i],'rb')
    #dn = pickle.load(F)
    #dn = dn.loc[:, ["StockID", "date", "min", "buy1", "sale1", "open", "close"]]
    #d1 = pd.concat([d1, dn], axis=0, ignore_index=True)
#d1.to_pickle('E:\\stock_minute_Rest.pkl')

# 2. calcualte alpha
F = open('E:\\INDEX_MIN_TICK_SH000300.pkl','rb')
index300 = pickle.load(F)
F = open('E:\\INDEX_MIN_TICK_SH000852.pkl','rb')
indexelse = pickle.load(F)
F = open('E:\\INDEX_MIN_TICK_SH000905.pkl','rb')
index500 = pickle.load(F)
F1 = open('E:\\stock_minute_IC.pkl','rb')
stock_IC = pickle.load(F1)
F2 = open('E:\\stock_minute_IF.pkl','rb')
stock_IF = pickle.load(F2)
F3 = open('E:\\stock_minute_CSI.pkl','rb')
F4 = open('E:\\stock_minute_CSIRest.pkl','rb')
F5 = open('E:\\stock_minute_Rest.pkl','rb')
stock_CSI = pickle.load(F3)
stock_CSIRest = pickle.load(F4)
stock_Rest = pickle.load(F5)
F1 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_CSI_60d.pkl",'rb')
F2 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_IF_60d.pkl",'rb')
F3 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\stockBeta_L_IC_60d.pkl",'rb')
sto_ck = pd.read_csv('E:\\all_stock.csv', encoding = "GBK").iloc[:,1:]
CSI_beta = pickle.load(F1)
IF_beta = pickle.load(F2)
IC_beta = pickle.load(F3)
csi_index_500_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_comp_SH000905.csv',
                        encoding = "utf-8")
csi_index_300_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000300.csv',
                            encoding = "utf-8")
csi_index_1000_com = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 3 event study\\index_comp_SH000852.csv',
                             encoding = "utf-8")
csi_index_500 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 1 lead-lag\\index_daily_SH000905.csv',
                            encoding = "utf-8").iloc[:,1:]
csi_index_300 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\index_daily_SH000300.csv',
                            encoding = "utf-8").iloc[:,1:]
csi_index_1000 = pd.read_csv('C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\index_daily_SH000852.csv',
                             encoding = "utf-8").iloc[:,1:]

def cal_return(df):
    df.loc[:, 'buy1'] = df['buy1'].replace(0.0, np.nan)
    df.loc[:, 'sale1'] = df['sale1'].replace(0.0, np.nan)
    df.loc[:, 'mid'] = df[['buy1', 'sale1']].mean(axis=1)
    df.loc[:, 'return'] = df.groupby("StockID")["mid"].apply(lambda x: x/x.shift(1)-1)
    return 0
def cal_return1(df):
    df.loc[:, 'return'] = df.loc[:, 'close'] / df.loc[:, 'close'].shift(1) - 1
    return 0

dff = pd.DataFrame()
date = stock_IF[(stock_IF["date"] >= 20190102) & (stock_IF["date"] <= 20191101)]["date"].unique()
# date = np.insert(date, 0, csi_index_500_com["Date"][csi_index_500_com.index[csi_index_500_com["Date"] == 20190102] - 1].values)
for i in range(1, len(date)):
    df = pd.DataFrame()

    tt1 = stock_IF[stock_IF["date"] == date[i]]
    tt2 = stock_IC[stock_IC["date"] == date[i]]
    tt3 = stock_CSI[stock_CSI["date"] == date[i]]
    tt4 = stock_CSIRest[stock_CSIRest["date"] == date[i]]
    tt5 = stock_Rest[stock_Rest["date"] == date[i]]
    ttall = pd.concat([tt1, tt2, tt3, tt4, tt5]).drop_duplicates().reset_index(drop=True)
    #p1 = pd.DataFrame(
        #csi_index_300_com.columns[(csi_index_300_com[csi_index_300_com["Date"] == date[i]] != 0).values[0]][1:],
        #columns=["Symbol"])
    #p2 = pd.DataFrame(
        #csi_index_500_com.columns[(csi_index_500_com[csi_index_500_com["Date"] == date[i]] != 0).values[0]][1:],
        #columns=["Symbol"])
    #p3 = pd.DataFrame(
        #list(set(ttall["StockID"].unique()) - set(p1["Symbol"].unique()) - set(p2["Symbol"].unique())),
        #columns=["Symbol"])
    #test1 = pd.merge(ttall, p1, left_on="StockID", right_on="Symbol", how="inner").iloc[:, :-1]
    #test2 = pd.merge(ttall, p2, left_on="StockID", right_on="Symbol", how="inner").iloc[:, :-1]
    #test3 = pd.merge(ttall, p3, left_on="StockID", right_on="Symbol", how="inner").iloc[:, :-1]
    test1 = ttall[ttall["StockID"].isin(csi_index_300_com.columns[(csi_index_300_com[csi_index_300_com["Date"]
                                                                                     == date[i]] != 0).values[0]][1:])]
    test2 = ttall[ttall["StockID"].isin(csi_index_500_com.columns[(csi_index_500_com[csi_index_500_com["Date"]
                                                                                     == date[i]] != 0).values[0]][1:])]
    test3 = ttall[ttall["StockID"].isin(list(set(ttall["StockID"].unique()) - set(test1["StockID"].unique())
                                             - set(test2["StockID"].unique())))]
    test1 = test1.sort_values(by=["StockID", "min"])
    test1 = test1.reset_index(drop=True)
    test2 = test2.sort_values(by=["StockID", "min"])
    test2 = test2.reset_index(drop=True)
    test3 = test3.sort_values(by=["StockID", "min"])
    test3 = test3.reset_index(drop=True)
    index1 = index300[index300["date"] == date[i]]
    index1 = pd.concat([index300[index300["date"] == date[i - 1]].tail(1), index1])
    index2 = index500[index500["date"] == date[i]]
    index2 = pd.concat([index500[index500["date"] == date[i - 1]].tail(1), index2])
    index3 = indexelse[indexelse["date"] == date[i]]
    index3 = pd.concat([indexelse[indexelse["date"] == date[i - 1]].tail(1), index3])

    cal_return(test1)
    cal_return1(index1)
    index1 = index1.iloc[1:, :]
    df1 = pd.merge(test1, index1, left_on="min", right_on="min", how="left").loc[:,
          ["date_x", "StockID", "min", "return_x", "return_y", "close_y"]]
    beta1 = pd.DataFrame(IF_beta.loc[date[i - 1], :])
    beta1["Symbol"] = beta1.index
    beta1.columns = ['beta', 'Symbol']
    df1 = pd.merge(df1, beta1, left_on="StockID", right_on="Symbol", how="left")
    df1["beta"] = df1["beta"].fillna(1.0)
    df_1 = pd.DataFrame(
        {"Date": df1["date_x"].values, 'ID': df1["StockID"].values, "min": df1["min"].values,
         "index": np.repeat("SH000300", len(df1)), "beta": df1["beta"].values, "stock_return": df1["return_x"].values,
         "index_return": df1["return_y"].values, "index_close": df1["close_y"]})

    cal_return(test2)
    cal_return1(index2)
    index2 = index2.iloc[1:, :]
    df2 = pd.merge(test2, index2, left_on="min", right_on="min", how="left").loc[:,
          ["date_x", "StockID", "min", "return_x", "return_y", "close_y"]]
    beta2 = pd.DataFrame(IC_beta.loc[date[i - 1], :])
    beta2["Symbol"] = beta2.index
    beta2.columns = ['beta', 'Symbol']
    df2 = pd.merge(df2, beta2, left_on="StockID", right_on="Symbol", how="left")
    df2["beta"] = df2["beta"].fillna(1.0)
    df_2 = pd.DataFrame(
        {"Date": df2["date_x"].values, 'ID': df2["StockID"].values, "min": df2["min"].values,
         "index": np.repeat("SH000905", len(df2)), "beta": df2["beta"].values,
         "stock_return": df2["return_x"].values, "index_return": df2["return_y"].values,
         "index_close": df2["close_y"]})

    cal_return(test3)
    cal_return1(index3)
    index3 = index3.iloc[1:, :]
    df3 = pd.merge(test3, index3, left_on="min", right_on="min", how="left").loc[:,
          ["date_x", "StockID", "min", "return_x", "return_y", "close_y"]]
    beta3 = pd.DataFrame(CSI_beta.loc[date[i - 1], :])
    beta3["Symbol"] = beta3.index
    beta3.columns = ['beta', 'Symbol']
    df3 = pd.merge(df3, beta3, left_on="StockID", right_on="Symbol", how="left")
    df3["beta"] = df3["beta"].fillna(1.0)
    df_3 = pd.DataFrame(
        {"Date": df3["date_x"].values, 'ID': df3["StockID"].values, "min": df3["min"].values,
         "index": np.repeat("SH000852", len(df3)), "beta": df3["beta"].values,
         "stock_return": df3["return_x"].values, "index_return": df3["return_y"].values,
         "index_close": df3["close_y"]})

    def final(df1, df11, test, index):
        stock1 = sto_ck[sto_ck["Date"] == date[i]]
        stock2 = sto_ck[sto_ck["Date"] == date[i - 1]]
        df1["open"] = pd.merge(df1, stock1, left_on="ID", right_on="Symbol", how="left")["open"]
        df1["close"] = pd.merge(df1, stock2, left_on="ID", right_on="Symbol", how="left")["close"]
        df1.loc[df1.groupby("ID")["stock_return"].head(1).index, "stock_return"] = (test.groupby("StockID")["mid"].
                                                      first().reset_index().iloc[:, 1].values
                                                      - df1.groupby("ID").head(1)["close"].values) \
                                                     / df1.groupby("ID").head(1)["close"].values
        df11["stock_return"] = pd.Series(
            (test.groupby("StockID")["mid"].first().reset_index().iloc[:, 1].values -
                df1.groupby("ID").head(1)["open"].values) / df1.groupby("ID").head(1)[
                "open"].values)
        df11["idx_return"] = pd.Series(
            (df1.groupby("ID")["index_close"].first().reset_index().iloc[:, 1].values - np.repeat(
                index[index["Date"] == date[i]]["open"], len(df11)).values) / np.repeat(
                index[index["Date"] == date[i]]["open"],
                len(df11)).values)
        df11["beta"] = pd.Series(df1.groupby("ID")["beta"].first().reset_index().iloc[:, 1].values)
        df11["index"] = pd.Series(df1.groupby("ID")["index"].first().reset_index().iloc[:, 1].values)
        df11["alpha_open21min"] = df11["stock_return"] - df11["beta"] * df11["idx_return"]
    dff_1 = df_1.groupby("ID").head(1).reset_index().loc[:, ["Date", "ID"]]
    final(df_1, dff_1, test1, csi_index_300)
    dff_2 = df_2.groupby("ID").head(1).reset_index().loc[:, ["Date", "ID"]]
    final(df_2, dff_2, test2, csi_index_500)
    dff_3 = df_3.groupby("ID").head(1).reset_index().loc[:, ["Date", "ID"]]
    final(df_3, dff_3, test3, csi_index_1000)

    df = pd.concat([df, df_1, df_2, df_3], ignore_index=True)
    df["alpha"] = df["stock_return"] - df["beta"]*df["index_return"]
    df = df.sort_values(by=["Date", "ID"])
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, 'E:\\alpha\\alpha_min_' + str(date[i]) + '.parquet')

    dff = pd.concat([dff, dff_1, dff_2, dff_3])

    print(df)
    print(dff)

dff = dff.sort_values(by=["Date", "ID"])
table = pa.Table.from_pandas(dff, preserve_index=False)
pq.write_table(table, 'E:\\alpha_open21min.parquet')

