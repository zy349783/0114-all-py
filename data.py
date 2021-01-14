import tushare as ts
import pandas as pd

token ='8722319c0d4b316258408cf41d9e781fa5e3cbb4695e4e2e492636e8'
ts.set_token(token)
pro = ts.pro_api()

df = pd.DataFrame()
list = pro.fund_basic(market='O')["ts_code"]
for i in list:
    df1 = pro.fund_portfolio(ts_code=i)
    df = pd.concat([df, df1])
    print(df)
df.to_pickle(r"E:\\公募基金持仓数据2.pkl")
print(df)

# df = pd.DataFrame()
# dff = pd.DataFrame()
# data = pro.trade_cal(exchange='', start_date='20190101', end_date='20191226')
# date = data[data["is_open"] == 1]["cal_date"].values
# for i in date:
#     df1 = pro.top_list(trade_date=i)
#     dff1 = pro.top_inst(trade_date=i)
#     df = pd.concat([df, df1])
#     dff = pd.concat([dff, dff1])
#     df.to_csv("E:\\龙虎榜每日明细.csv", encoding="GBK")
#     dff.to_csv("E:\\龙虎榜机构明细.csv", encoding="GBK")
