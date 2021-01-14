import pandas as pd
import numpy as np

def ff(x):
    ts = np.unique(x.iloc[:,0])
    db = x[x.iloc[:,0] == ts[0]].iloc[:,[1,2]]
    for i in range(1,len(ts)):
        dd = x[x.iloc[:,0] == ts[i]].iloc[:,[1,2]]
        db = pd.merge(dd, db, how="outer", left_on="Date", right_on="Date")
    db.columns = ts
    return db

da_ta = pd.read_csv('E:\\beta.csv', encoding="utf-8").loc[:,["Symbol","Date","s_returns","Industry"]]
#x = da_ta.groupby("Industry")["Symbol", "Date", "s_returns"].apply(ff)

te_st = pd.concat([da_ta[da_ta["Industry"] == "SW801010"], da_ta[da_ta["Industry"] == "SW801020"]])
x = te_st.groupby("Industry")["Symbol", "Date", "s_returns"].apply(ff)
print(x)