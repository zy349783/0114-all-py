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

df = pd.read_csv("E:\\ST_table.csv", encoding="utf-8").iloc[:, 1:]
# me_an = []
# for i in range(len(df)):
#     try:
#         me_an.append(np.nanmean(eval(df.loc[i, "X21"], {'nan': float('nan')})))
#         print(i)
#     except:
#         print(df.loc[i, :])
# df["mean"] = pd.Series(me_an)
# df.to_csv()

from sklearn.linear_model import LogisticRegression
df.dropna(inplace=True)
X_train = df.loc[df["Date"] < 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
X_test = df.loc[df["Date"] > 20180000, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16",
               "X17", "mean"]]
y_train = df.loc[df["Date"] < 20180000, ["Y"]]
y_test = df.loc[df["Date"] > 20180000, ["Y"]]
y_train = y_train["Y"]
X1 = pd.concat([X_train, y_train], axis=1)
ST = pd.read_csv("C:\\Users\\win\\Downloads\\ST_effectdate.csv", encoding="utf-8")

# 1. Generate Synthetic Samples + logistic regression
def SMOTE_logit(r):
    os = SMOTE(sampling_strategy=r)
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_y = pd.DataFrame(os_data_y.values, columns=['y'])
    # Recursive Feature Elimination
    from sklearn.feature_selection import RFE
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)
    cols = os_data_X.columns[rfe.support_]
    X = os_data_X[cols]
    y = os_data_y['y']
    # Model Implementation
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(method='bfgs')
    # print(result.summary2())
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = os_data_X[cols]
    # Logistic regression
    logreg = LogisticRegression(class_weight="balanced")
    logreg.fit(X, y)
    y_pred = logreg.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # print("Precision:", metrics.precision_score(y_test, y_pred))
    # print("Recall:", metrics.recall_score(y_test, y_pred))
    dff = pd.concat([y_test.reset_index().iloc[:, 1:], pd.DataFrame(y_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
          "ID"].reset_index().iloc[:, 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    # print("ratio of corrected predicted stocks:", r1)
    # print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred),
            metrics.accuracy_score(y_test, y_pred), r1, r2]

# 2. Generate Synthetic Samples + decision trees
def SMOTE_DT(r):
    os = SMOTE(sampling_strategy=r)
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_y = pd.DataFrame(os_data_y.values, columns=['y'])
    # Recursive Feature Elimination
    from sklearn.feature_selection import RFE
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)
    cols = os_data_X.columns[rfe.support_]
    X = os_data_X[cols]
    y = os_data_y['y']
    # Model Implementation
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(method='bfgs')
    # print(result.summary2())
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = os_data_X[cols]
    # Logistic regression
    rfc = RandomForestClassifier(n_estimators=100).fit(X, y)
    rfc_pred = rfc.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
    # print(cnf_matrix)
    # print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))
    # print("Precision:", metrics.precision_score(y_test, rfc_pred))
    # print("Recall:", metrics.recall_score(y_test, rfc_pred))
    dff = pd.concat([y_test.reset_index().iloc[:, 1:], pd.DataFrame(rfc_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
     "ID"].reset_index().iloc[:, 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred),
            metrics.accuracy_score(y_test, rfc_pred), r1, r2]

# 3. Oversampling Minority Class + logistic regression
def Over_logit(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    ST_upsampled = resample(ST, replace=True, n_samples=len(not_ST), random_state=27)
    upsampled = pd.concat([not_ST, ST_upsampled])
    y_train = upsampled["Y"]
    X_train = upsampled.drop(["Y"], axis=1)
    # Model Implementation
    import statsmodels.api as sm
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(method='bfgs')
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = X_train[cols]
    y = y_train
    # Logistic regression
    logreg = LogisticRegression(class_weight="balanced")
    logreg.fit(X, y)
    y_pred = logreg.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    dff = pd.concat([y_test.reset_index().iloc[:, 1:], pd.DataFrame(y_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
          "ID"].reset_index().iloc[:, 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    # print("ratio of corrected predicted stocks:", r1)
    # print("ratio of prediction accuracy:", r2)
    return [metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred),
            metrics.accuracy_score(y_test, y_pred), r1, r2]

# 4. Oversampling Minority Class + decision trees
def Over_DT(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    ST_upsampled = resample(ST, replace=True, n_samples=len(not_ST), random_state=27)
    upsampled = pd.concat([not_ST, ST_upsampled])
    y_train = upsampled["Y"]
    X_train = upsampled.drop(["Y"], axis=1)
    # Model Implementation
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(method='bfgs')
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = X_train[cols]
    y = y_train
    # Logistic regression
    rfc = RandomForestClassifier(n_estimators=100).fit(X, y)
    rfc_pred = rfc.predict(X_test[cols])
    cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
    dff = pd.concat([y_test.reset_index().iloc[:, 1:], pd.DataFrame(rfc_pred, columns=["Y1"]), df.loc[df["Date"] > 20180000,
          "ID"].reset_index().iloc[:, 1:]], axis=1)
    list1 = dff[dff["Y"] == 1]["ID"].unique()
    list2 = dff[dff["Y1"] == 1]["ID"].unique()
    list3 = dff[(dff["Y"] == 1) & (dff["Y1"] == 1)]["ID"].unique()
    r1 = len(set(list1) & set(list3)) / len(list1)
    r2 = len(set(list1) & set(list3)) / len(list2)
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred),
            metrics.accuracy_score(y_test, rfc_pred), r1, r2]

# 5. Undersampling + logistic regression
def Under_logit(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    not_ST_downsampled = resample(not_ST, replace=True, n_samples=len(ST), random_state=27)
    downsampled = pd.concat([not_ST_downsampled, ST])
    y_train = downsampled["Y"]
    X_train = downsampled.drop(["Y"], axis=1)
    # Model Implementation
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(method='bfgs')
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = X_train[cols]
    y = y_train
    # Logistic regression
    logreg = LogisticRegression(solver='liblinear').fit(X, y)
    y_pred = logreg.predict(X_test[cols])
    return [metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.accuracy_score(y_test, y_pred)]

# 6. Undersampling + logistic decision trees
def Under_DT(X):
    not_ST = X[X["Y"] == 0]
    ST = X[X["Y"] == 1]
    not_ST_downsampled = resample(not_ST, replace=True, n_samples=len(ST), random_state=27)
    downsampled = pd.concat([not_ST_downsampled, ST])
    y_train = downsampled["Y"]
    X_train = downsampled.drop(["Y"], axis=1)
    # Model Implementation
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(method='bfgs')
    cols = result.pvalues[result.pvalues < 0.05].index.values
    X = X_train[cols]
    y = y_train
    # Logistic regression
    rfc = RandomForestClassifier(n_estimators=100).fit(X, y)
    rfc_pred = rfc.predict(X_test[cols])
    return [metrics.precision_score(y_test, rfc_pred), metrics.recall_score(y_test, rfc_pred), metrics.accuracy_score(y_test, rfc_pred)]

df1 = pd.DataFrame()
re1 = Over_logit(X1)
df1 = df1.append(pd.DataFrame({"Ratio": 0.5, "Precision": re1[0], "Recall": re1[1], "Accuracy": re1[2],
                               "Method": "Over_logit", "Ratio1": re1[3], "Ratio2": re1[4]}, index=[1]), ignore_index=True)
re2 = Over_DT(X1)
df1 = df1.append(pd.DataFrame({"Ratio": 0.5, "Precision": re2[0], "Recall": re2[1], "Accuracy": re2[2],
                               "Method": "Over_DT", "Ratio1": re2[3], "Ratio2": re2[4]}, index=[1]), ignore_index=True)
# cols = [] under this condition, all variables p value >= 0.05
# re3 = Under_logit(X1)
# df1 = df1.append(pd.DataFrame({"Ratio": 0.5, "Precision": re3[0], "Recall": re3[1], "Method": "Under_logit"},
#                                   index=[1]), ignore_index=True)
# re4 = Under_DT(X1)
# df1 = df1.append(pd.DataFrame({"Ratio": 0.5, "Precision": re4[0], "Recall": re4[1], "Method": "Under_DT"},
#                                   index=[1]), ignore_index=True)
print(df1)
for i in np.arange(0.001, 0.009, 0.001):
    re = SMOTE_logit(i)
    df1 = df1.append(pd.DataFrame({"Ratio": i, "Precision": re[0], "Recall": re[1], "Accuracy": re[2],
                                   "Method": "SMOTE_logit", "Ratio1": re[3], "Ratio2": re[4]}, index=[i]), ignore_index=True)
for j in np.arange(0.001, 0.009, 0.001):
    re = SMOTE_DT(j)
    df1 = df1.append(pd.DataFrame({"Ratio": j, "Precision": re[0], "Recall": re[1], "Accuracy": re[2],
                                   "Method": "SMOTE_DT", "Ratio1": re[3], "Ratio2": re[4]}, index=[j]), ignore_index=True)

df1.to_csv("E:\\ST_methods.csv", encoding="utf-8")


















# final part: visualize the result
# create heatmap
# from sklearn import metrics
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# class_names = [0, 1]
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# annot_kws = {"ha": 'center', "va": 'top'}
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='d', annot_kws=annot_kws)
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.show()
# # model evaluation 2
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Recall:", metrics.recall_score(y_test, y_pred))
# # ROC curve
# y_pred_proba = logreg.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.show()


