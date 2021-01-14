import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import Formatter
import glob
from pandas.plotting import autocorrelation_plot

def cal(df):
    df["R2"] = df["R2"].rank(ascending=False)
    df = df[df["R2"] <= int(df.shape[0] / 10)]
    return df.loc[:, ["Date", "Paired_Stocks", "beta", "alpha"]]


class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos"""
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return pd.to_datetime(self.dates[ind], format="%Y%m%d").strftime(self.fmt)

class backtest:
    original_df = pd.DataFrame()
    forecast_df = pd.DataFrame()
    def __init__(self, df1, df2):
        self.original_df = df1
        self.forecast_df = df2
    def new_strategy(self):
        df = self.original_df
        re = self.forecast_df
        dff = pd.DataFrame(columns=["Date", "spread"])
        dff1 = pd.DataFrame(columns=["Date", "gp1", "gp2", "gp3", "gp4", "gp5"])
        date = re["Date"].unique()
        date1 = list(df["Date"].unique())
        for i in range(len(date)):
            ma_trix = re[re["Date"] == date[i]]
            ma_trix = ma_trix.groupby("Target").apply(cal).reset_index()
            alpha = pd.pivot_table(ma_trix, values="alpha", index=["Paired_Stocks"], columns=["Target"])
            beta = pd.pivot_table(ma_trix, values="beta", index=["Paired_Stocks"], columns=["Target"])
            a1 = df[df["Date"] == date[i]][df[df["Date"] == date[i]]["ID"].isin(alpha.index)]["alpha"]
            a1 = pd.DataFrame(a1.values, index=df[df["Date"] == date[i]][df[df["Date"] ==
                                                                            date[i]]["ID"].isin(alpha.index)]["ID"])
            prod = pd.concat([a1.mul(beta.iloc[:, i], axis=0) for i in range(beta.shape[1])], axis=1)
            prod.set_axis(beta.columns, axis=1, inplace=True)
            ret = prod + alpha
            ret = ret.loc[:, ret.std().sort_values()[: int(len(ret) / 2)].index.values]
            gp5 = ret.mean().sort_values().iloc[:int(ret.shape[1] / 5)].index.values
            gp4 = ret.mean().sort_values().iloc[int(ret.shape[1] / 5): 2 * int(ret.shape[1] / 5)].index.values
            gp3 = ret.mean().sort_values().iloc[2 * int(ret.shape[1] / 5): 3 * int(ret.shape[1] / 5)].index.values
            gp2 = ret.mean().sort_values().iloc[3 * int(ret.shape[1] / 5): 4 * int(ret.shape[1] / 5)].index.values
            gp1 = ret.mean().sort_values().iloc[4 * int(ret.shape[1] / 5):].index.values
            # for j in range(1, 6):
            #     spread = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                          df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp1)]["alpha"]) - \
            #              np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                          df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp5)]["alpha"])
            #     top = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                       df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp1)]["alpha"])
            #     bottom = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                          df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp5)]["alpha"])
            #     m2 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                      df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp2)]["alpha"])
            #     m3 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                      df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp3)]["alpha"])
            #     m4 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + j]][
            #                      df[df["Date"] == date1[date1.index(date[i]) + j]]["ID"].isin(gp4)]["alpha"])
            #     dff1 = dff1.append({"Date": date1[date1.index(date[i]) + j], "gp1": top, "gp2": m2, "gp3": m3,
            #                         "gp4": m4, "gp5": bottom}, ignore_index=True)
            #     dff = dff.append({"Date": date1[date1.index(date[i]) + j], "spread": spread, "Top": top,
            #                       "Bottom": bottom}, ignore_index=True)
            spread = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                                 df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp1)]["alpha"]) - \
                     np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                                 df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp5)]["alpha"])
            top = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                              df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp1)]["alpha"])
            bottom = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                                 df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp5)]["alpha"])
            m2 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                             df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp2)]["alpha"])
            m3 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                             df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp3)]["alpha"])
            m4 = np.mean(df[df["Date"] == date1[date1.index(date[i]) + 1]][
                             df[df["Date"] == date1[date1.index(date[i]) + 1]]["ID"].isin(gp4)]["alpha"])
            dff1 = dff1.append({"Date": date1[date1.index(date[i]) + 1], "gp1": top, "gp2": m2, "gp3": m3,
                                "gp4": m4, "gp5": bottom}, ignore_index=True)
            dff = dff.append({"Date": date1[date1.index(date[i]) + 1], "spread": spread, "Top": top,
                              "Bottom": bottom}, ignore_index=True)
        dff.to_csv("E:\\result.csv", encoding="utf-8")
        dff1.to_csv("E:\\result_all_groups.csv", encoding="utf-8")
        # dff.to_csv("E:\\result_after_delete.csv", encoding="utf-8")

        # dff = pd.read_csv("E:\\result.csv", encoding="utf-8").iloc[:, 1:]
        print(dff.describe())
        dff["Date"] = dff["Date"].apply(lambda x: str(x)[:8])
        dff["spread"] = dff["spread"] * 10000
        dff["Top"] = dff["Top"] * 10000
        dff["Bottom"] = dff["Bottom"] * 10000
        mean = np.mean(dff["spread"])
        var = np.var(dff["spread"])
        T = mean * np.sqrt(len(dff)) / np.nanstd(dff["spread"])
        m1 = np.mean(dff["Top"])
        m2 = np.mean(dff["Bottom"])
        T1 = m1 * np.sqrt(len(dff)) / np.nanstd(dff["Top"])
        T2 = m2 * np.sqrt(len(dff)) / np.nanstd(dff["Bottom"])
        var1 = np.var(dff["Top"])
        var2 = np.var(dff["Bottom"])

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(np.arange(len(dff)), dff["spread"],
                color='blue', alpha=2, linewidth=1, linestyle='-', marker='.', markersize=2, label='daily spread')
        ax.plot(np.arange(len(dff)), dff["Top"],
                color='red', alpha=0.5, linewidth=1, linestyle='-.', marker='.', markersize=2, label='top group alpha')
        ax.plot(np.arange(len(dff)), dff["Bottom"],
                color='green', alpha=0.5, linewidth=1, linestyle='-.', marker='.', markersize=2,
                label='bottom group alpha')
        ax.set_xlabel('')
        ax.set_ylabel('bps')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(MyFormatter(dff["Date"].values, '%Y%m%d'))
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)

        textstr = '\n'.join((
            r'spread mean = %.5f' % mean, r'spread variance = %.5f' % var, r'spread T value = %.5f' % T,
            r'top group mean = %.5f' % m1, r'top group variance = %.5f' % var1, r'top group T value = %.5f' % T1,
            r'bottom group mean = %.5f' % m2, r'bottom group variance = %.5f' % var2,
            r'bottom group T value = %.5f' % T2,
        ), )
        props = dict(facecolor='red', alpha=0.25, pad=10)
        ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, horizontalalignment='left')

        ax.set_title("Daily spread result for strategy based on alpha forecast")
        ax.grid()
        plt.show()


# df = pd.read_csv('E:\\daily_alpha.csv', encoding="utf-8").iloc[:, 1:]
# df = df.sort_values(by=["Date", "ID"])
# re = pd.read_csv('E:\\final_result.csv', encoding="utf-8").iloc[:, 1:]
# test1 = backtest(df, re)
# test1.new_strategy()

path = r'E:\final result'
all_files = glob.glob(path + "/*.csv")
re = pd.read_csv(all_files[0], encoding="GBK").iloc[:, 1:]
for i in range(1, len(all_files)):
    dn = pd.read_csv(all_files[i], encoding="GBK").iloc[:, 1:]
    re = pd.concat([re, dn], axis=0, ignore_index=True)
df = pd.read_csv('E:\\daily_alpha.csv', encoding="utf-8").iloc[:, 1:]
df = df.sort_values(by=["Date", "ID"])
test1 = backtest(df, re)
test1.new_strategy()


dff = pd.read_csv("E:\\result.csv", encoding="utf-8").iloc[:, 1:]
autocorrelation_plot(dff["spread"])
plt.show()

nsteps = 200
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
autocorrelation_plot(walk)
plt.show()
