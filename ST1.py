import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns

ST1 = pd.read_csv("C:\\Users\\win\\Downloads\\ST_effectdate.csv", encoding="utf-8")
ST1["StockID"] = ST1["StockID"].str[:8]
ST1.drop_duplicates(inplace=True)
ST1 = ST1.reset_index().iloc[:, 1:]
ST2 = pd.read_csv("C:\\Users\\win\\Downloads\\ST_financialReport.csv", encoding="utf-8")
F1 = open(r'E:\\SZ_announce.pkl', 'rb')
F2 = open(r'E:\\SH_announce.pkl', 'rb')
SZ = pickle.load(F1)
SH = pickle.load(F2)
test1 = pd.merge(ST1, ST2, left_on=["year", "StockID"], right_on=["ST_year", "StockID"], how="left")
ST = test1.loc[np.isnan(test1["ST_year"]), ["StockID", "date", "year", "month"]]
data = pd.read_csv("E:\\all_stock.csv", encoding="GBK").iloc[:, 1:]
alpha = pd.read_csv('E:\\new_beta1.csv', encoding="utf-8").iloc[:, 1:]
ST3 = pd.read_csv("C:\\Users\\win\\Downloads\\STlist (1).csv", encoding="utf-8")

def func(M, N):
    date = np.sort(data["Date"].unique())
    date = date[::-1]
    df1 = pd.DataFrame()
    for t in range(M, len(date)-N):
        slist = data[data["Date"] == date[t]]["Symbol"].values
        slist = list(set(slist) - set(ST3.columns[(ST3[ST3["index"] == date[t]] == 0).any()].values))
        df = pd.DataFrame()
        df["ID"] = pd.Series(slist)
        df["Date"] = date[t]
        df["Y"] = 0
        df["X1"] = 0
        df["X2"] = 0
        df["X3"] = 0
        df["X4"] = 0
        df["X5"] = 0
        df["X6"] = 0
        df["X7"] = 0
        df["X8"] = 0
        df["X9"] = 0
        df["X10"] = 0
        df["X11"] = 0
        df["X12"] = 0
        df["X13"] = 0
        df["X14"] = 0
        df["X15"] = 0
        df["X16"] = 0
        df["X17"] = 0
        df["X18"] = 0
        df["X19"] = 0
        df["X20"] = 0

        ST_list = ST[(ST["date"] <= date[t-M]) & (ST["date"] > date[t])]["StockID"].values
        df.loc[df["ID"].isin(ST_list), "Y"] = 1
        shs = df.loc[df["ID"].str[:2] =='SH', "ID"].values
        szs = df.loc[df["ID"].str[:2] =='SZ', "ID"].values
        data1 = SH[(SH["StockID"].isin(shs)) & (SH["date"] < date[t]) & (SH["date"] >= date[t+N])]
        data2 = SZ[(SZ["StockID"].isin(szs)) & (SZ["date"] < date[t]) & (SZ["date"] >= date[t+N])]
        # 1. 立案调查、中国证券监督管理委员会调查通知书
        l1 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(立案调查|中国证券监督管理委员会'
                                                           r'调查通知书)\S+').str.len() != 0]["StockID"].values)
        l1 = np.append(l1, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(立案调查|中国证券'
                                                 r'监督管理委员会调查通知书)\S+').str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l1), "X1"] = 1
        # 2. 延期披露
        l2 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+延期披露\S+报告\S+').str.len()
                             != 0]["StockID"].values)
        l2 = np.append(l2, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+延期披露\S+报告\S+').str.len()
                             != 0]["StockID"].values))
        df.loc[df["ID"].isin(l2), "X2"] = 1
        # 3. 业绩快报、业绩预告修正
        l3 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(业绩快报修正|业绩预告修正)\S+').str.len()
                             != 0]["StockID"].values)
        l3 = np.append(l3, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(业绩快报修正|业绩预告修正)'
                                                                         r'\S+').str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l3), "X3"] = 1
        # 4. 计提资产减值
        l4 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+计提资产减值\S+').str.len()
                             != 0]["StockID"].values)
        l4 = np.append(l4, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+计提资产减值\S+').str.len()
                             != 0]["StockID"].values))
        df.loc[df["ID"].isin(l4), "X4"] = 1
        # 5. 证监局
        l5 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+证监局\S+').str.len()
                             != 0]["StockID"].values)
        l5 = np.append(l5, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+证监局\S+').str.len()
                             != 0]["StockID"].values))
        df.loc[df["ID"].isin(l5), "X5"] = 1
        # 6.股份-冻结
        l6 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+股份(?!.*解除)\S+冻结\S+').str.len()
                             != 0]["StockID"].values)
        l6 = np.append(l6, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+股份(?!.*解除)\S+冻结\S+')
                                     .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l6), "X6"] = 1
        # 7. 诉讼
        l7 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(涉诉|诉讼)\S+').str.len()
                             != 0]["StockID"].values)
        l7 = np.append(l7, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(涉诉|诉讼)\S+').str.len()
                             != 0]["StockID"].values))
        df.loc[df["ID"].isin(l7), "X7"] = 1
        # 8. 关注函，问询函
        l8 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(关注函|问询函)\S+').str.len()
                             != 0]["StockID"].values)
        l8 = np.append(l8, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(关注函|问询函)\S+')
                                     .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l8), "X8"] = 1
        # 9. 监管工作函
        l9 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+监管工作函\S+').str.len()
                             != 0]["StockID"].values)
        l9 = np.append(l9, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+监管工作函\S+').str.len()
                             != 0]["StockID"].values))
        df.loc[df["ID"].isin(l9), "X9"] = 1
        # 10. 对外担保、资金占用
        l10 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(对外担保\S+独立意见|违规担保|'
                                                            r'非经营性资金占用|关联方占用|关联方资金占用|自查对外担保事项|'
                                                            r'对外担保事项未披露)').str.len() != 0]["StockID"].values)
        l10 = np.append(l10, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(对外担保\S+独立意见|'
                                                            r'违规担保|非经营性资金占用|关联方占用|关联方资金占用|'
                                                            r'自查对外担保事项|对外担保事项未披露)').str.len() != 0]
                                                            ["StockID"].values))
        df.loc[df["ID"].isin(l10), "X10"] = 1
        # 11. 无法按期归还募集资金
        l11 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+无法按期归还募集资金').str.len()
                              != 0]["StockID"].values)
        l11 = np.append(l11, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+无法按期归还募集资金')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l11), "X11"] = 1
        # 12. 债务逾期
        l12 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+债务逾期').str.len()
                              != 0]["StockID"].values)
        l12 = np.append(l12, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+债务逾期')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l12), "X12"] = 1
        # 13. 到期未兑付/未按期兑付
        l13 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(到期未兑付|未按期兑付)').str.len()
                              != 0]["StockID"].values)
        l13 = np.append(l13, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(到期未兑付|未按期兑付)')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l13), "X13"] = 1
        # 14. 总经理/董事长/董事会秘书/监事/高级管理人员/独立董事辞职
        l14 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(总经理辞职|董事长辞职|董事会秘书辞职|'
                                                            r'监事辞职|高级管理人员辞职|独立董事辞职)').str.len()
                                                            != 0]["StockID"].values)
        l14 = np.append(l14, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+(总经理辞职|董事长辞职|'
                                                                           r'董事会秘书辞职|监事辞职|高级管理人员辞职|'
                                                                           r'独立董事辞职)').str.len()
                                                                           != 0]["StockID"].values))
        df.loc[df["ID"].isin(l14), "X14"] = 1
        # 15. 风险警示提示
        l15 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+风险警示\S+提示').str.len()
                              != 0]["StockID"].values)
        l15 = np.append(l15, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+风险警示\S+提示')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l15), "X15"] = 1
        # 16. 银行账户-冻结
        l16 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+银行账户\S+冻结').str.len()
                              != 0]["StockID"].values)
        l16 = np.append(l16, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+银行账户\S+冻结')
                                       .str.len() != 0]["StockID"].values))

        df.loc[df["ID"].isin(l16), "X16"] = 1
        # 17. 停产
        l17 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+停产').str.len() != 0]["StockID"].values)
        l17 = np.append(l17, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+停产')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l17), "X17"] = 1
        # 18. 无法在法定期限内披露定期报告
        l18 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+无法在法定期限内披露定期报告')
                        .str.len() != 0]["StockID"].values)
        l18 = np.append(l18, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+无法在法定期限内'
                                                                           r'披露定期报告').str.len() != 0]
                                                                           ["StockID"].values))
        df.loc[df["ID"].isin(l18), "X18"] = 1
        # 19. 债务到期未获清偿
        l19 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+债务到期未获清偿')
                        .str.len() != 0]["StockID"].values)
        l19 = np.append(l19, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+债务到期未获清偿')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l19), "X19"] = 1
        # 20. 新增资产查封
        l20 = np.unique(data1[data1["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+资产查封')
                        .str.len() != 0]["StockID"].values)
        l20 = np.append(l20, np.unique(data2[data2["contents"].str.findall(r'^[\u4e00-\u9fa5]\S+资产查封')
                                       .str.len() != 0]["StockID"].values))
        df.loc[df["ID"].isin(l20), "X20"] = 1
        # 21. alpha
        print(df.loc[(df.iloc[:, 2:] == 1).sum(1) >= 4, :])
        alpha1 = alpha[(alpha["Symbol"].isin(slist)) & (alpha["Date"] < date[t]) & (alpha["Date"] >= date[t + N])]
        alpha1 = alpha1.pivot(index="ID", columns="Date", values="alpha")
        # alpha1["X21"] = alpha1.values.tolist()
        # alpha1 = alpha1.reset_index()
        alpha1["X21"] = alpha1.mean(axis=1)
        alpha1 = alpha1.reset_index()
        df = pd.merge(df, alpha1.loc[:, ["ID", "X21"]], left_on="ID", right_on="ID")
        df1 = pd.concat([df1, df])
        print(df1)
    df1.to_csv("E:\\ST_table1.csv", encoding="utf-8")

func(20, 60)
