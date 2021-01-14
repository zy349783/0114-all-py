import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns

def find_ST(x):
    li_st = np.where(x != x.shift(1))[0]
    li_st = list(li_st[1:])
    for i in li_st:
        if (x[i-1] == 1) & (x[i] == 0):
            continue
        else:
            li_st.remove(i)
    if len(li_st) != 0:
        data["index"][li_st]

# 1. calculate beta with respect to corresponding index
# sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
# csi_index_500 = pd.read_csv('C:\\Users\\win\\Downloads\\index_daily_SH000905.csv',
#                             encoding = "utf-8").iloc[:,1:]
# csi_index_300 = pd.read_csv('C:\\Users\\win\\Downloads\\index_daily_SH000300.csv',
#                             encoding = "utf-8").iloc[:,1:]
# csi_index_1000 = pd.read_csv('C:\\Users\\win\\Downloads\\index_daily_SH000852.csv',
#                              encoding = "utf-8").iloc[:,1:]
# csi_index_500_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000905.csv',
#                         encoding = "utf-8")
# csi_index_300_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000300.csv',
#                             encoding = "utf-8")
# csi_index_1000_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000852.csv',
#                              encoding = "utf-8")
# start_date = max(csi_index_300["Date"][0], csi_index_500["Date"][0], csi_index_1000["Date"][0], min(sto_ck["Date"]))
# end_date = min(csi_index_300["Date"].iloc[-1], csi_index_500["Date"].iloc[-1], csi_index_1000["Date"].iloc[-1],
#                max(sto_ck["Date"]))
# sto_ck["returns"] = sto_ck["ret"]/100
# csi_index_500["returns"] = csi_index_500['close']/csi_index_500['close'].shift(1)-1
# csi_index_300["returns"] = csi_index_300['close']/csi_index_300['close'].shift(1)-1
# csi_index_1000["returns"] = csi_index_1000['close']/csi_index_1000['close'].shift(1)-1
# sto_ck = sto_ck.sort_values(by=["Date", "Symbol"])
# df = pd.DataFrame(columns=["Date", "ID", "index", "stock_return", "index_return"])
# date = sto_ck[(sto_ck["Date"] >= start_date) & (sto_ck["Date"] <= end_date)]["Date"].unique()
#
# for i in range(len(date)):
#     te_st = sto_ck[sto_ck["Date"] == date[i]]
#     stocks = te_st["Symbol"]
#     csi300 = pd.DataFrame(csi_index_300_com.columns[(csi_index_300_com[csi_index_300_com["Date"] == date[i]] != 0).values[0]][1:], columns=["Symbol"])
#     csi500 = pd.DataFrame(csi_index_500_com.columns[(csi_index_500_com[csi_index_500_com["Date"] == date[i]] != 0).values[0]][1:], columns=["Symbol"])
#     csiothers = pd.DataFrame(list(set(stocks) - set(csi300["Symbol"]) - set(csi500["Symbol"])), columns=["Symbol"])
#
#     df1 = pd.merge(csi300, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
#     df_1 = pd.DataFrame(
#         {"Date": np.repeat(date[i], len(df1)), 'ID': df1["Symbol"].values, "index": np.repeat("SH000300", len(df1)),
#          "stock_return": df1["returns"].values, "index_return": np.repeat(csi_index_300[csi_index_300["Date"] ==
#          date[i]]["returns"].values[0], len(df1))})
#
#     df2 = pd.merge(csi500, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
#     df_2 = pd.DataFrame(
#         {"Date": np.repeat(date[i], len(df2)), 'ID': df2["Symbol"].values, "index": np.repeat("SH000905", len(df2)),
#          "stock_return": df2["returns"].values, "index_return": np.repeat(csi_index_500[csi_index_500["Date"] ==
#          date[i]]["returns"].values[0], len(df2))
#          })
#
#     df3 = pd.merge(csiothers, te_st, left_on="Symbol", right_on="Symbol", how="inner")[["Symbol", "returns"]]
#     df_3 = pd.DataFrame(
#         {"Date": np.repeat(date[i], len(df3)), 'ID': df3["Symbol"].values, "index": np.repeat("SH000852", len(df3)),
#          "stock_return": df3["returns"].values, "index_return": np.repeat(csi_index_1000[csi_index_1000["Date"] ==
#          date[i]]["returns"].values[0], len(df3))
#          })
#
#     df = pd.concat([df, df_1, df_2, df_3])
#     print(df)
#
# df = df.sort_values(by=["ID", "Date"])
# df.to_csv("E:\\stock_index_ret.csv", encoding="utf-8")
# def cal(x):
#     x.columns = ["x", "y"]
#     x_cov = x.rolling(60).cov().unstack()['x']['y']
#     x_var = x['y'].to_frame().rolling(60).var()
#     result = x_cov / x_var.iloc[:, 0]
#     re = pd.DataFrame(result.values, index=range(0, len(result)))
#     print(re)
#     return re
# df["beta"] = df.groupby("ID")['stock_return', 'index_return'].apply(cal).reset_index().iloc[:, 2]
# df["beta"] = df.groupby("ID")["beta"].shift(1)
# df["alpha"] = df["stock_return"] - df["beta"]*df["index_return"]
# df.to_csv('E:\\new_beta1.csv', encoding="utf-8")

# alpha = pd.read_csv('E:\\new_beta1.csv', encoding="utf-8").iloc[:, 1:]
# sto_ck = pd.read_csv('E:\\all_stock.csv', encoding="GBK").iloc[:, 1:]
# df = pd.merge(alpha, sto_ck.loc[:, ["Date", "Symbol", "open", "close"]], left_on=["Date", "ID"],
#               right_on=["Date", "Symbol"])
# df.to_csv('E:\\new_beta1.csv', encoding="utf-8")




data = pd.read_csv("E:\\STlist.csv", encoding="utf-8")
alpha = pd.read_csv('E:\\new_beta1.csv', encoding="utf-8").iloc[:, 1:]
alpha["ret_open2close"] = alpha["close"]/alpha["open"] - 1
alpha["ret"] = (alpha["stock_return"] + 1)/(alpha["ret_open2close"] + 1) - 1
ST = pd.read_csv("C:\\Users\\win\\Downloads\\ST_effectdate.csv", encoding="utf-8")
da_te = alpha["Date"]
data1 = data.loc[:, ~(data != 0).all()]
data1 = data1.reset_index().iloc[:, 1:]
ST1 = pd.DataFrame()
date = ST["date"].unique()
for i in date:
    ST2 = pd.DataFrame()
    ST2[i] = pd.Series(ST[ST["date"] == i]["StockID"].values)
    ST1 = pd.concat([ST1, ST2], axis=1)
print(ST1)


def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]

def cal2(dataframe):
    df = pd.DataFrame()
    st = pd.DataFrame()
    CAR = []
    T = []
    m = 0
    for i in dataframe.columns:
        stocks = dataframe[i][~dataframe[i].isnull()]
        for j in stocks:
            if "." in j:
                j = j[:8]
            data1 = alpha[alpha["ID"] == j]
            da_te = data1["Date"]
            event_date = i
            if len(da_te[da_te == event_date]) == 0:
                if len(da_te[da_te > event_date]) == 0:
                    print(event_date)
                    print(j)
                    continue
                else:
                    event_date = da_te[da_te > event_date].iloc[0]
                # print(event_date)
                # print(j)
                # continue
            time_period = da_te.loc[da_te[da_te == event_date].index[0] - 10:da_te[da_te == event_date].index[0] + 10]
            tpp = pd.DataFrame(time_period)
            # tp = pd.DataFrame(pd.merge(tpp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            tp = pd.DataFrame(pd.merge(tpp, data1, left_on="Date", right_on="Date", how="left")["stock_return"])
            tp1 = pd.DataFrame(pd.merge(tpp, data1, left_on="Date", right_on="Date", how="left")["alpha"])
            if (tp.isna().all().values[0]) | (tp1.isna().all().values[0]):
            # if tp.isna().all().values[0]:
                print(event_date)
                print(j)
                continue
            else:
                tp = tp.rename(columns={'alpha': m})
                df = pd.concat([df, tp], axis=1)
                data2 = data1[data1["Date"] >= event_date]
                fall = data2["stock_return"][data2["stock_return"] < -0.045]
                if len(fall) == 0:
                    times = 0
                    first = 0
                else:
                    if data2.loc[fall.index[0], "Date"] != event_date:
                        times = 0
                        first = 0
                    else:
                        times = 1
                        first = 1
                        last_date = event_date
                        for i1 in fall.index:
                            if i1 + 1 in fall.index:
                                times = times + 1
                                last_date = data2.loc[i1 + 1, "Date"]
                            else:
                                break
                if data2.loc[data2["Date"] == event_date, "ret"].values[0] < -0.045:
                    first1 = 1
                else:
                    first1 = 0
                t1 = data1[data1["Date"] < event_date].iloc[-1, 0]
                t2 = data1[data1["Date"] < event_date].iloc[-5, 0]
                t3 = data1[data1["Date"] < event_date].iloc[-10, 0]
                st1 = pd.DataFrame({'首日': event_date, '总跌停': times, '首日跌停': first, '开盘跌停': first1,
                                    "末日": last_date, "提前1日": t1, "提前5日": t2, "提前10日": t3}, index=[j])
                st = st.append(st1)
                m = m + 1
    print(df)
    CAR = np.cumsum(df.mean(axis=1)) * 10000
    T = df.mean(axis=1) * 10000
    M = df.median(axis=1) * 10000
    st.to_csv("E:\\跌停统计1.csv", encoding="GBK")
    return pd.DataFrame({"CAR": CAR, "T value": T, "Median_alpha": M, "mean": np.mean(st["总跌停"]),
                         "median": np.median(st["总跌停"]), "prob": "{:.1%}".format(np.sum(st["首日跌停"])/len(st)),
                         "prob1": "{:.1%}".format(np.sum(st["开盘跌停"])/len(st)), "n": m})

x = range(-10, 11)
sns.set()
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

r1 = cal2(ST1)
mean = r1["mean"].values[0]
median = r1["median"].values[0]
prob = r1["prob"].values[0]
prob1 = r1["prob1"].values[0]
l = r1["n"].values[0]
ax1.axvspan(-10, 0, facecolor='lightgreen', alpha=0.5)
ax1.axvspan(0, 10, facecolor='green', alpha=0.5)
ax1.set_ylabel('Cumulative Return', fontname="Arial", fontsize=8)
ax1.plot(x, r1["CAR"], marker='.', color='blue', alpha=1, linewidth=1, markersize=2)
ax1.tick_params('y')
ax1.tick_params(labelsize=8)
ax1.axhline(y=0, color='salmon', linestyle=(0, (1, 1)), alpha=0.6, linewidth=1)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Return', fontname="Arial", fontsize=8)
ax2.bar(x, r1["T value"], color='salmon', alpha=0.8)
ax2.scatter(x, r1["Median_alpha"], color='red', alpha=0.8, s=4)
ax2.tick_params('y')
ax2.tick_params(labelsize=8)
plt.title("All ST stocks cumulative return before and after effect day", fontname="Arial", fontsize=10)
ax1.grid(True)
ax2.grid(None)
textstr = '\n'.join((
            r'mean of continuous fall = %.2f' % mean, r'median of continuous fall = %.2f' % median,
            r'prob of first day fall = ' + prob, r'total number of stocks = %d' % l,
            r'prob of first day open fall = ' + prob1,
        ), )
props = dict(facecolor='red', alpha=0.25, pad=10)
ax1.text(0.03, 0.2, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, horizontalalignment='left')
align_yaxis_np(ax1, ax2)

fig.tight_layout()
plt.show()

# re2 = pd.read_csv("E:\\跌停统计1.csv", encoding="GBK")
# re1 = pd.read_csv("E:\\跌停统计.csv", encoding="GBK")
# fig, ax = plt.subplots()
# sns.distplot(re1["总跌停"][re1["总跌停"] != 0], bins=max(re1["总跌停"])+1, hist_kws={"range": [0, max(re1["总跌停"])+1]})
# print(len(re1["总跌停"]))
# print(len(re1["总跌停"][re1["总跌停"] != 0]))
# plt.xlabel("cumulative fall days")
# plt.title("distribution of cumulative fall days for gp1 stocks")
# ax.grid()
# plt.show()
# fig, ax = plt.subplots()
# sns.distplot(re2["总跌停"][re2["总跌停"] != 0], bins=max(re2["总跌停"])+1, hist_kws={"range": [0, max(re2["总跌停"])+1]})
# print(len(re2["总跌停"]))
# print(len(re2["总跌停"][re2["总跌停"] != 0]))
# plt.xlabel("cumulative fall days")
# plt.title("distribution of cumulative fall days for gp2 stocks")
# ax.grid()
# plt.show()

# csi_index_500_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000905.csv',
#                         encoding = "utf-8")
# csi_index_300_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000300.csv',
#                             encoding = "utf-8")
# csi_index_1000_com = pd.read_csv('C:\\Users\\win\\Downloads\\index_comp_SH000852.csv',
#                              encoding = "utf-8")
# ST = pd.read_csv("C:\\Users\\win\\Downloads\\ST_effectdate.csv", encoding="utf-8")
# date = ST["date"].unique()
# ST1 = pd.DataFrame()
# df = pd.DataFrame()
# for i in date:
#     ST2 = pd.DataFrame()
#     ST2[i] = pd.Series(ST[ST["date"] == i]["StockID"].values)
#     ST1 = pd.concat([ST1, ST2], axis=1)
# print(ST1)
# m = 0
# for i in ST1.columns:
#     stocks = ST1[i][~ST1[i].isnull()]
#     for j in stocks:
#         if "." in j:
#             j = j[:8]
#         if j in csi_index_300_com.columns:
#             if csi_index_300_com[csi_index_300_com["Date"] == i][j].values[0] != 0:
#                 df1 = pd.DataFrame({'Date': i, 'Stock': j, 'Index': "IF"}, index=[m])
#                 df = df.append(df1, ignore_index=True)
#         if j in csi_index_500_com.columns:
#             if csi_index_500_com[csi_index_500_com["Date"] == i][j].values[0] != 0:
#                 df1 = pd.DataFrame({'Date': i, 'Stock': j, 'Index': "IC"}, index=[m])
#                 df = df.append(df1, ignore_index=True)
#         if j in csi_index_1000_com.columns:
#             if csi_index_1000_com[csi_index_1000_com["Date"] == i][j].values[0] != 0:
#                 df1 = pd.DataFrame({'Date': i, 'Stock': j, 'Index': "CSI1000"}, index=[m])
#                 df = df.append(df1, ignore_index=True)
#         if len(df[(df["Date"] == i) & (df["Stock"] == j)]) == 0:
#             df1 = pd.DataFrame({'Date': i, 'Stock': j, 'Index': "CSI Rest"}, index=[m])
#             df = df.append(df1, ignore_index=True)
#         m = m + 1
# df.to_csv("E:\\ST_index.csv", encoding="utf-8")

# df = pd.read_csv("E:\\ST_index.csv", encoding="utf-8")
# df["Date"] = df["Date"].apply(lambda x: str(x))
# print(df)

df = pd.read_csv(r"C:\Users\win\Desktop\work\project 8 ST prediction\event_desc.csv", encoding="utf-8")
print(df)