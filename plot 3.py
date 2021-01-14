import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import resample
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE

df = pd.read_csv("E:\\ST_table.csv", encoding="utf-8").iloc[:, 1:]
ST = pd.read_csv("C:\\Users\\win\\Downloads\\ST_effectdate.csv", encoding="utf-8")
re1 = pd.read_csv("C:\\Users\\win\\Desktop\\work\\project 8 ST prediction\\ST\\ST_methods_1.csv", encoding="utf-8")
re2 = pd.read_csv("C:\\Users\\win\\Desktop\\work\\project 8 ST prediction\\ST\\ST_methods_2.csv", encoding="utf-8")
df.dropna(inplace=True)
X_train = df.loc[df["Date"] < 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
X_test = df.loc[df["Date"] > 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
y_train = df.loc[df["Date"] < 20180000, ["Y"]]
y_test = df.loc[df["Date"] > 20180000, ["Y"]]
y_train = y_train["Y"]
X1 = pd.concat([X_train, y_train], axis=1)

dff1 = re1[re1["Method"] == "SMOTE_logit"]
x = dff1["Ratio"]
y = dff1["Recall"]
z = dff1["Precision"]
k = dff1["Accuracy"]
ax = plt.subplot(111)
x = np.arange(len(dff1))
ax.bar(x-0.3, y.values, width=0.3, color='b', label='Recall')
ax.bar(x, z.values, width=0.3, color='g', label='Precision')
ax.bar(x+0.3, k.values, width=0.3, color='r', label="Accuracy")
plt.xticks(x, dff1["Ratio"].values)
plt.xlabel("Ratio")
plt.title("Plot of SMOTE-logit result under different SMOTE ratio")
plt.legend()
plt.show()

dff2 = re1[re1["Method"] == "SMOTE_DT"]
dff2 = dff2.sort_values(by=["Ratio"])
x = dff2["Ratio"]
y = dff2["Recall"]
z = dff2["Precision"]
k = dff2["Accuracy"]
ax = plt.subplot(111)
x = np.arange(len(dff2))
ax.bar(x-0.3, y.values, width=0.3, color='b', label='Recall')
ax.bar(x, z.values, width=0.3, color='g', label='Precision')
ax.bar(x+0.3, k.values, width=0.3, color='r', label="Accuracy")
plt.xticks(x, dff2["Ratio"].values)
plt.xlabel("Ratio")
plt.title("Plot of SMOTE-RandomForrest result under different SMOTE ratio")
plt.legend()
plt.show()

dff1 = re2[re2["Method"] == "SMOTE_logit"]
dff1 = dff1.sort_values(by=["Ratio"])
x = dff1["Ratio"]
y = dff1["Recall"]
z = dff1["Precision"]
k = dff1["Accuracy"]
l = dff1["Ratio1"]
m = dff1["Ratio2"]
ax = plt.subplot(111)
x = np.arange(len(dff1))
ax.bar(x-0.2, y.values, width=0.1, color='b', label='Recall')
ax.bar(x-0.1, z.values, width=0.1, color='g', label='Precision')
ax.bar(x, k.values, width=0.1, color='r', label="Accuracy")
ax.bar(x+0.1, y.values, width=0.1, color='y', label='Ratio1')
ax.bar(x+0.2, z.values, width=0.1, color='c', label='Ratio2')
plt.xticks(x, dff1["Ratio"].values.round(3))
plt.xlabel("Ratio")
plt.title("Plot of SMOTE-balanced_weight_logit result under different SMOTE ratio")
plt.legend()
plt.show()

dff1 = re2[re2["Method"] == "SMOTE_NB"]
dff1 = dff1.sort_values(by=["Ratio"])
x = dff1["Ratio"]
y = dff1["Recall"]
z = dff1["Precision"]
k = dff1["Accuracy"]
l = dff1["Ratio1"]
m = dff1["Ratio2"]
ax = plt.subplot(111)
x = np.arange(len(dff1))
ax.bar(x-0.2, y.values, width=0.1, color='b', label='Recall')
ax.bar(x-0.1, z.values, width=0.1, color='g', label='Precision')
ax.bar(x, k.values, width=0.1, color='r', label="Accuracy")
ax.bar(x+0.1, y.values, width=0.1, color='y', label='Ratio1')
ax.bar(x+0.2, z.values, width=0.1, color='c', label='Ratio2')
plt.xticks(x, dff1["Ratio"].values.round(3))
plt.xlabel("Ratio")
plt.title("Plot of SMOTE-NB result under different SMOTE ratio")
plt.legend()
plt.show()