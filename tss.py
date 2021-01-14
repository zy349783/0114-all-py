import tushare as ts
import pandas as pd
import numpy as np
import re
import glob
import pickle
from twisted.internet import task, reactor
from datetime import datetime
import schedule
import time

#da_te = pd.read_csv('E:\\dateList.csv', encoding="utf-8")
#da_te = da_te.rename(columns={'19901219': 'Date'})
token='8722319c0d4b316258408cf41d9e781fa5e3cbb4695e4e2e492636e8'
ts.set_token(token)
pro = ts.pro_api()
#d1 = da_te[(da_te['Date']>=20100104) & (da_te['Date']<=20170102)]

#df = pd.DataFrame()
#path = r'E:\use'
#all_files = glob.glob(path + "/*.csv")
#for i in range(3):
    #dd = pd.read_csv(all_files[i], encoding = "GBK")
    #da_te = all_files[0][18:26]
    #li_st = np.unique(dd[dd["trade_stats"] == 1].iloc[:, 1])
    #for i in range(len(li_st)):
        #li_st[i] = re.split('(\d+)', li_st[i])[1] + '.' + re.split('(\d+)', li_st[i])[0]
    #d1 = pro.moneyflow(trade_date=da_te)
    #da_ta = d1.loc[d1["ts_code"].isin(li_st)]
    #da_ta.rename(columns={"ts_code": "StockID", "trade_date": "date"}, inplace=True)
    #da_ta.loc[:, "StockID"] = pd.DataFrame(da_ta["StockID"].apply(lambda x: x.split('.')[1] + x.split('.')[0]))
    #df = df.append(da_ta, ignore_index=True)


#for i in range(len(d1)):
    #db = pro.moneyflow(trade_date=str(d1.iloc[i,:][0]))
    #db.rename(columns={"ts_code": "StockID", "trade_date": "date"}, inplace=True)
    #db.loc[:, "StockID"] = pd.DataFrame(db["StockID"].apply(lambda x: x.split('.')[1] + x.split('.')[0]))
    #db.to_csv("E:\\data\\"+str(d1.iloc[i,:][0])+".csv", encoding="utf-8")

def cal_moneyflow():
    date = datetime.today().strftime('%Y%m%d')
    db = pro.moneyflow(trade_date=date)
    if db.empty == False:
        print("Data collected now!!!!!!!!!!!", datetime.now())
        old1 = pd.read_csv('E:\\MoneyFlow.csv', encoding="utf-8").iloc[:, 1:]
        db.rename(columns={"ts_code": "StockID", "trade_date": "date"}, inplace=True)
        db.loc[:, "StockID"] = pd.DataFrame(db["StockID"].apply(lambda x: x.split('.')[1] + x.split('.')[0]))
        newcolumn = db.columns
        old1 = old1.reindex(columns=newcolumn)
        old1 = pd.concat([old1, db])
        old1 = old1.sort_values(by=["StockID", "date"])
        old1.to_csv("E:\\MoneyFlow.csv", encoding="utf-8")
        print("data saved!!!!!!!!!!!", datetime.now())
        # reactor.stop()
    else:
        print("No data collected yet", datetime.now())

if __name__ == '__main__':
    cal_moneyflow()
# def loop():
#     timeout = 60.0
#     l = task.LoopingCall(cal_moneyflow)
#     l.start(timeout)
#     reactor.run()
#
# schedule.every().day.at("17:11").do(loop)
# while True:
#     schedule.run_pending()
#     time.sleep(1)
#
#
#
# #path = r'E:\data'
# #all_files = glob.glob(path + "/*.csv")
# #dd = pd.DataFrame()
# #for i in range(len(all_files)):
#      #dn = pd.read_csv(all_files[i], encoding="utf-8").iloc[:,1:]
#      #dd = dd.append(dn)
# #dd = dd.sort_values(by=["StockID","date"])
# #dd.to_csv('E:\\MoneyFlow.csv', encoding="utf-8")
#
#
#
