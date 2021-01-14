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
df.dropna(inplace=True)
X_train = df.loc[df["Date"] < 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
X_test = df.loc[df["Date"] > 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
y_train = df.loc[df["Date"] < 20180000, ["Y"]]
y_test = df.loc[df["Date"] > 20180000, ["Y"]]
y_train = y_train["Y"]
X1 = pd.concat([X_train, y_train], axis=1)

def feature(X_train, y_train):
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X_train, y_train.values.ravel())
    cols = X_train.columns[rfe.support_]
    X = X_train[cols]
    y = y_train
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(method='bfgs')
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = X[cols]
    y = y
    return [X, y]

def Over(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    ST_upsampled = resample(ST, replace=True, n_samples=len(not_ST), random_state=27)
    upsampled = pd.concat([not_ST, ST_upsampled])
    y_train = upsampled["Y"]
    X_train = upsampled.drop(["Y"], axis=1)
    return [X_train, y_train]

def Under(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    not_ST_downsampled = resample(not_ST, replace=False, n_samples=len(ST), random_state=27)
    downsampled = pd.concat([not_ST_downsampled, ST])
    y_train = downsampled["Y"]
    X_train = downsampled.drop(["Y"], axis=1)
    return [X_train, y_train]

def SMOTE1(r):
    os = SMOTE(sampling_strategy=r)
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_y = pd.DataFrame(os_data_y.values, columns=['y'])
    return [os_data_X, os_data_y]

def logit(X, y):
    cols = X.columns.values
    logreg = LogisticRegression(class_weight="balanced")
    # logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    dff = pd.concat(
        [y_test.reset_index().iloc[:, 1:], pd.DataFrame(y_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000, "ID"]
                                                                                   .reset_index().iloc[:, 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    print("ratio of corrected predicted stocks:", r1)
    print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred),
            metrics.accuracy_score(y_test, y_pred), r1, r2]

def DT(X, y):
    cols = X.columns.values
    rfc = RandomForestClassifier(n_estimators=10, class_weight="balanced").fit(X, y)
    # rfc = RandomForestClassifier(n_estimators=10).fit(X, y)
    rfc_pred = rfc.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))
    print("Precision:", metrics.precision_score(y_test, rfc_pred))
    print("Recall:", metrics.recall_score(y_test, rfc_pred))
    dff = pd.concat(
        [y_test.reset_index().iloc[:, 1:], pd.DataFrame(rfc_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
                                                                                          "ID"].reset_index().iloc[:,
                                                                                   1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    print("ratio of corrected predicted stocks:", r1)
    print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred),
            metrics.accuracy_score(y_test, rfc_pred), r1, r2]

def SVM(X, y):
    cols = X.columns.values
    svclassifier = SVC(kernel='linear', class_weight="balanced")
    # svclassifier = SVC(kernel='linear')
    svclassifier.fit(X, y)
    y_pred = svclassifier.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    dff = pd.concat(
        [y_test.reset_index().iloc[:, 1:], pd.DataFrame(y_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
                                                                                        "ID"].reset_index().iloc[:,
                                                                                 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    print("ratio of corrected predicted stocks:", r1)
    print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred),
            metrics.accuracy_score(y_test, y_pred), r1, r2]

def NB(X, y):
    cols = X.columns.values
    model = GaussianNB()
    rfc_pred = model.fit(X, y).predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))
    print("Precision:", metrics.precision_score(y_test, rfc_pred))
    print("Recall:", metrics.recall_score(y_test, rfc_pred))
    dff = pd.concat(
        [y_test.reset_index().iloc[:, 1:], pd.DataFrame(rfc_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
                                                                                          "ID"].reset_index().iloc[:,
                                                                                   1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    print("ratio of corrected predicted stocks:", r1)
    print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred),
            metrics.accuracy_score(y_test, rfc_pred), r1, r2]

def GB(X, y):
    cols = X.columns.values
    model = GradientBoostingClassifier()
    rfc_pred = model.fit(X, y).predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))
    print("Precision:", metrics.precision_score(y_test, rfc_pred))
    print("Recall:", metrics.recall_score(y_test, rfc_pred))
    dff = pd.concat(
        [y_test.reset_index().iloc[:, 1:], pd.DataFrame(rfc_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
                                                                                          "ID"].reset_index().iloc[:,
                                                                                   1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    print("ratio of corrected predicted stocks:", r1)
    print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred),
            metrics.accuracy_score(y_test, rfc_pred), r1, r2]

df1 = pd.DataFrame()

for i in np.arange(0.01, 0.1, 0.01):
    re = SMOTE1(i)
    re = feature(re[0], re[1])
    re = GB(re[0], re[1])
    df1 = df1.append(pd.DataFrame({"Ratio": i, "Precision": re[0], "Recall": re[1], "Accuracy": re[2],
                                   "Method": "SMOTE_GB", "Ratio1": re[3], "Ratio2": re[4]}, index=[i]), ignore_index=True)
for i in np.arange(0.001, 0.01, 0.001):
    re = SMOTE1(i)
    re = feature(re[0], re[1])
    re = GB(re[0], re[1])
    df1 = df1.append(pd.DataFrame({"Ratio": i, "Precision": re[0], "Recall": re[1], "Accuracy": re[2],
                                   "Method": "SMOTE_GB", "Ratio1": re[3], "Ratio2": re[4]}, index=[i]), ignore_index=True)
for i in np.arange(0.1, 1, 0.1):
    re = SMOTE1(i)
    re = feature(re[0], re[1])
    re = GB(re[0], re[1])
    df1 = df1.append(pd.DataFrame({"Ratio": i, "Precision": re[0], "Recall": re[1], "Accuracy": re[2],
                                   "Method": "SMOTE_GB", "Ratio1": re[3], "Ratio2": re[4]}, index=[i]), ignore_index=True)
df1.to_csv("E:\\ST_methods.csv", encoding="utf-8")
# re = feature(X_train, y_train)
# # logit(re[0], re[1])
# # DT(re[0], re[1])
# NB(re[0], re[1])
# GB(re[0], re[1])