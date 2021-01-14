import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta
import calendar
import seaborn as sns
import glob
from datetime import datetime

def create_name(x):
    return str(x.iloc[0]) + '_' + str(x.iloc[1])

def get_series(ID, day, min, length):
    data1 = pd.read_parquet('E:\\alpha\\alpha_min_' + day + '.parquet', engine='pyarrow')
    min_data1 = data1[data1["ID"] == ID].loc[:, ["Date", "min", "alpha"]].dropna()
    min_data1.reset_index(inplace=True)
    close2open = pd.read_parquet('E:\\alpha_LD1close2open.parquet', engine='pyarrow')
    open21min = pd.read_parquet('E:\\alpha_open21min.parquet', engine='pyarrow')
    path = r'E:\alpha'
    all_files = glob.glob(path + "/*.parquet")
    i1 = all_files.index('E:\\alpha\\alpha_min_' + day + '.parquet')
    data2 = pd.read_parquet(all_files[i1 - 1], engine='pyarrow')
    min_data2 = data2[data2["ID"] == ID].loc[:, ["Date", "min", "alpha"]].dropna()
    min_data2.reset_index(inplace=True)
    data3 = pd.read_parquet(all_files[i1 + 1], engine='pyarrow')
    min_data3 = data3[data3["ID"] == ID].loc[:, ["Date", "min", "alpha"]].dropna()
    min_data3.reset_index(inplace=True)

    in_dex = min_data1.index[min_data1["min"] == min][0]
    if in_dex > length:
        tp = pd.Series(min_data1.loc[in_dex-length:in_dex-1, "alpha"].values, index=
        min_data1.loc[in_dex-length:in_dex-1, ["Date", "min"]].transpose().apply(create_name).values)
    elif in_dex == length:
        tp = pd.Series(open21min[(open21min["Date"] == int(day)) & (open21min["ID"] == ID)]
                       ["alpha_open21min"].values, index=["open21min"])
        tp = tp.append(pd.Series(min_data1.loc[1:in_dex-1, "alpha"].values, index=
        min_data1.loc[1:in_dex-1, ["Date", "min"]].transpose().apply(create_name).values))
    elif in_dex == length-1:
        tp = pd.Series(close2open[(close2open["Date"] == int(day)) & (close2open["ID"] == ID)]
                       ["alphaLD1close_open"].values, index=["close2open"])
        tp = tp.append(pd.Series(open21min[(open21min["Date"] == int(day)) & (open21min["ID"] == ID)]
                                 ["alpha_open21min"].values, index=["open21min"]))
        tp = tp.append(pd.Series(min_data1.loc[1:in_dex-1, "alpha"].values, index=
        min_data1.loc[1:in_dex-1, ["Date", "min"]].transpose().apply(create_name).values))
    else:
        tp = pd.Series(min_data2[in_dex-length+1:]["alpha"].values,
                       index=min_data2[in_dex-length+1:].loc[:, ["Date", "min"]].transpose().apply(create_name).values)
        tp = tp.append(pd.Series(close2open[(close2open["Date"] == int(day)) & (close2open["ID"] == ID)]
                                 ["alphaLD1close_open"].values, index=["close2open"]))
        tp = tp.append(pd.Series(open21min[(open21min["Date"] == int(day)) & (open21min["ID"] == ID)]
                                 ["alpha_open21min"].values, index=["open21min"]))
        tp = tp.append(pd.Series(min_data1.loc[1:in_dex-1, "alpha"].values, index=
        min_data1.loc[1:in_dex-1, ["Date", "min"]].transpose().apply(create_name).values))

    if in_dex <= min_data1.index[-1] - length:
        tp = tp.append(pd.Series(min_data1.loc[in_dex:in_dex+length, "alpha"].values, index=
        min_data1.loc[in_dex:in_dex+length, ["Date", "min"]].transpose().apply(create_name).values))
    elif in_dex == min_data1.index[-1] - length + 1:
        tp = tp.append(pd.Series(min_data1.loc[in_dex:, "alpha"].values, index=
        min_data1.loc[in_dex:, ["Date", "min"]].transpose().apply(create_name).values))
        tp = tp.append(pd.Series(close2open[(close2open["Date"] == int(all_files[i1 + 1][19:27])) &
                                (close2open["ID"] == ID)]["alphaLD1close_open"].values, index=["close2open"]))
    elif in_dex == min_data1.index[-1] - length + 2:
        tp = tp.append(pd.Series(min_data1.loc[in_dex:, "alpha"].values, index=
        min_data1.loc[in_dex:, ["Date", "min"]].transpose().apply(create_name).values))
        tp = tp.append(pd.Series(close2open[(close2open["Date"] == int(all_files[i1 + 1][19:27])) &
                                (close2open["ID"] == ID)]["alphaLD1close_open"].values, index=["close2open"]))
        tp = tp.append(pd.Series(open21min[(open21min["Date"] == int(all_files[i1 + 1][19:27])) &
                                (open21min["ID"] == ID)]["alpha_open21min"].values, index=["open21min"]))
    else:
        tp = tp.append(pd.Series(min_data1.loc[in_dex:, "alpha"].values, index=
        min_data1.loc[in_dex:, ["Date", "min"]].transpose().apply(create_name).values))
        tp = tp.append(pd.Series(close2open[(close2open["Date"] == int(all_files[i1 + 1][19:27])) &
                                (close2open["ID"] == ID)]["alphaLD1close_open"].values, index=["close2open"]))
        tp = tp.append(pd.Series(open21min[(open21min["Date"] == int(all_files[i1 + 1][19:27])) &
                                (open21min["ID"] == ID)]["alpha_open21min"].values, index=["open21min"]))
        tp = tp.append(pd.Series(min_data3.loc[1:in_dex-min_data1.index[-1]+length-2, "alpha"].values, index=
        min_data3.loc[1:in_dex-min_data1.index[-1]+length-2, ["Date", "min"]].transpose().apply(create_name).values))
    return tp

# 20190104 - 20191031
print(get_series("SH600000", "20190401", 236, 10))

