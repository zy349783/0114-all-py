import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

#test = pd.read_parquet('E:\\alpha_LD1close2open.parquet', engine='pyarrow')
#test1 = pd.read_parquet('C:\\Users\\win\\Downloads\\alpha_LD1close2open_Lun.parquet', engine='pyarrow')
#test2 = pd.read_parquet('C:\\Users\\win\\Downloads\\alpha_LD1close2open.parquet', engine='pyarrow')
#F1 = open("C:\\Users\\win\\Desktop\\work\\project 4 event study - minute\\alpha_LD1close2open.pkl",'rb')
#alpha1 = pickle.load(F1)
#alpha1 = alpha1.sort_values(by=["date","StockID"])
#test2 = test2[test2["Date"] > 20160301]
#test1 = test1[(test1["date"] <= 20191101) & (test1["date"] > 20160301)]
#test = test[test["Date"] >= 20190102]
#new_df = pd.merge(test1, test, left_on=["date", "StockID"], right_on=["Date", "ID"])
#new_df['diff'] = abs(new_df["alpha_LD1close2open"] - new_df["alphaLD1close_open"])
#print(new_df["diff"].describe())

#test2 = pd.read_parquet('E:\\alpha_open21min.parquet', engine='pyarrow')
#print(test2)


# 1. test MoneyFlow
#sto_ck = pd.read_csv('E:\\MoneyFlow.csv', encoding = "GBK")
#sto_ck1 = pd.read_csv('C:\\Users\\win\\Downloads\\MoneyFlow (1).csv', encoding = "GBK")

# solution 1
#df = sto_ck1.merge(sto_ck, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
#print(df)

# solution 2
# df = pd.concat([sto_ck, sto_ck1])
# df = df.reset_index(drop = True)
# df_gpby = df.groupby(list(df.columns))
# idx = [x[0] for x in df_gpby.groups.values() if len(x) != 1]
# df.reindex(idx)

# solution 3
#df1 = pd.merge(sto_ck, sto_ck1, left_on=["StockID", "date"], right_on=["StockID", "date"], how="right")
#print(abs(df1["buy_sm_vol_x"]-df1["buy_sm_vol_y"]).describe())
#print(abs(df1["buy_sm_amount_x"]-df1["buy_sm_amount_y"]).describe())
#print(abs(df1["sell_sm_vol_x"]-df1["sell_sm_vol_y"]).describe())
#print(abs(df1["sell_sm_amount_x"]-df1["sell_sm_amount_y"]).describe())
#print(abs(df1["buy_md_vol_x"]-df1["buy_md_vol_y"]).describe())
#print(abs(df1["buy_md_amount_x"]-df1["buy_md_amount_y"]).describe())
#print(abs(df1["sell_md_vol_x"]-df1["sell_md_vol_y"]).describe())
#print(abs(df1["sell_md_amount_x"]-df1["sell_md_amount_y"]).describe())
#print(abs(df1["buy_lg_vol_x"]-df1["buy_lg_vol_y"]).describe())
#print(abs(df1["buy_lg_amount_x"]-df1["buy_lg_amount_y"]).describe())
#print(abs(df1["sell_lg_vol_x"]-df1["sell_lg_vol_y"]).describe())
#print(abs(df1["sell_lg_amount_x"]-df1["sell_lg_amount_y"]).describe())
#print(abs(df1["buy_elg_vol_x"]-df1["buy_elg_vol_y"]).describe())
#print(abs(df1["buy_elg_amount_x"]-df1["buy_elg_amount_y"]).describe())
#print(abs(df1["sell_elg_vol_x"]-df1["sell_elg_vol_y"]).describe())
#print(abs(df1["sell_elg_amount_x"]-df1["sell_elg_amount_y"]).describe())
#print(abs(df1["net_mf_vol_x"]-df1["net_mf_vol_y"]).describe())
#print(abs(df1["net_mf_amount_x"]-df1["net_mf_amount_y"]).describe())


# 2. test alpha_open21min
path1 = r'E:\alpha'
path2 = r'C:\Users\win\Downloads\AlphaDecomposition'
all_files1 = glob.glob(path1 + "/*.parquet")
all_files2 = glob.glob(path2 + "/*.parquet")
df = pd.DataFrame()
for i in range(1, len(all_files2)):
    test1 = pd.read_parquet(all_files1[i-1], engine='pyarrow')
    test2 = pd.read_parquet(all_files2[i], engine='pyarrow')
    new_df = pd.merge(test2, test1, left_on=["StockID", "min"], right_on=["ID", "min"])
    new_df['diff'] = abs(new_df["alpha_min"] - new_df["alpha"])
    df = pd.concat([df, new_df['diff']])
print(df.describe())



test1 = pd.read_parquet('E:\\alpha\\alpha_min_20190104.parquet', engine='pyarrow')
test2 = pd.read_parquet('C:\\Users\\win\\Downloads\\AlphaDecomposition\\alpha_min_20190104_Lun.parquet', engine='pyarrow')
test1 = test1[["Date", "ID", "min", "index", "beta", "alpha", "stock_return", "index_return", "open", "close"]]
new_df = pd.merge(test2, test1, left_on=["StockID", "min"], right_on=["ID", "min"])
new_df['diff'] = abs(new_df["alpha_min"] - new_df["alpha"])
print(new_df["diff"].describe())
print(test1)
F = open('E:\\stock\\IF\\MINDATA_from_TICK_SH000300_20190614.pkl','rb')
test3 = pickle.load(F)
print(test3)

# 3. test money flow data
test1 = pd.read_csv("E:\\MoneyFlow.csv", encoding="utf-8")
test2 = pd.read_csv("E:\\factor_cal.csv", encoding="utf-8").iloc[:, 1:]
test2.fillna(0, inplace=True)
test1 = test1[(test1["date"] >= 20190902) & (test1["date"] <= 20191122)].loc[:, ["StockID", "date",
                                "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
                                "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
                                "net_mf_amount"]]
new_df = pd.merge(test1, test2, left_on=["date", "StockID"], right_on=["date", "StockID"], how="right")
def diag(x):
    df = x.corr().iloc[1:10, 10:]
    return pd.DataFrame(np.diag(df), index=[df.index, df.columns], columns=["correlation"])
re1 = new_df.groupby("date").apply(diag).reset_index()
re2 = new_df.groupby("StockID").apply(diag).reset_index()
re1["date"] = re1["date"].apply(lambda x: str(x))
date_time = pd.to_datetime(re1["date"])
nre1 = pd.DataFrame()
nre1["buy_sm_amount"] = re1[re1["level_1"] == "buy_sm_amount_x"]["correlation"].values
nre1["sell_sm_amount"] = re1[re1["level_1"] == "sell_sm_amount_x"]["correlation"].values
nre1["buy_md_amount"] = re1[re1["level_1"] == "buy_md_amount_x"]["correlation"].values
nre1["sell_md_amount"] = re1[re1["level_1"] == "sell_md_amount_x"]["correlation"].values
nre1["buy_lg_amount"] = re1[re1["level_1"] == "buy_lg_amount_x"]["correlation"].values
nre1["sell_lg_amount"] = re1[re1["level_1"] == "sell_lg_amount_x"]["correlation"].values
nre1["buy_elg_amount"] = re1[re1["level_1"] == "buy_elg_amount_x"]["correlation"].values
nre1["sell_elg_amount"] = re1[re1["level_1"] == "sell_elg_amount_x"]["correlation"].values
nre1["net_mf_amount"] = re1[re1["level_1"] == "net_mf_amount_x"]["correlation"].values
nre1 = nre1.set_index(date_time.unique())
nre1.plot()
plt.xlabel("Date")
plt.ylabel("correlations")
plt.title("plot of correlation for each variable across dates")
plt.show()

sns.distplot(re2.groupby("StockID")["correlation"].mean())
plt.title("distribution of correlation across stocks")
plt.show()
print(new_df)

# re1 = pd.read_csv("E:\\跌停统计.csv", encoding="GBK")
# re2 = pd.read_csv("E:\\跌停统计1.csv", encoding="GBK")
# re1.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
# re2.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
# df = pd.merge(re1, re2, left_on="id", right_on="id", how="outer")
# sns.distplot(re1["跌停"].values, bins=max(re1["跌停"].values))
# plt.title("distribution of cumulative fall for gp1", fontname="Arial", fontsize=10)
# plt.xlabel('cumulative fall days', fontname="Arial", fontsize=8)
# plt.show()
# sns.distplot(re2["跌停"].values, bins=max(re2["跌停"].values))
# plt.title("distribution of cumulative fall for gp2", fontname="Arial", fontsize=10)
# plt.xlabel('cumulative fall days', fontname="Arial", fontsize=8)
# plt.show()
# print(re1)