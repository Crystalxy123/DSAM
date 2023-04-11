# Import the requried packages
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model, neighbors, tree, svm, ensemble, neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate

import numpy as np
from collections import Counter
from sklearn.utils import shuffle
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')
# ---数据预处理
df = pd.read_excel("data_DSAM.XLSX")
df1 = df[["Infection", "Government risk management efficiency", "Emergency preparedness",
          "Quality and Accessibility of Care Index", "education level",
          "percentage of population of low age", "Population density", "Mass living level", "Monitoring and diagnosis"]]

# dimension
dm1 = ["Government risk management efficiency", "Emergency preparedness",
       "Quality and Accessibility of Care Index", "Monitoring and diagnosis"]
dm2 = ["percentage of population of low age", "education level",
       "Population density", "Mass living level"]
dm3 = ["Infection"]

# 数据归一化
df_normal = (df1 - df1.min()) / (df1.max() - df1.min())

# 相似度计算
# KNN找邻居

# ##dimension3

df03 = df_normal[dm3]

list1 = df03.iloc[:, 0].to_list()
k = 9


def test(list_, k):
    # list_tmp = list_
    other = []
    for q in range(len(list_)):
        target_figure = list_[q]
        distances = {}
        for i in range(len(list_)):
            dis = {i: abs(target_figure - list_[i])}
            distances.update(dis)
        distances_sorted = sorted(distances.items(), key=lambda x: x[1], reverse=False)
        res = []
        cnt = 0
        for i in range(1, k):
            res.append(distances_sorted[i][0])
        other.append(res)
    return other


# visulize 距离图
nbs = test(list1, k)
print(nbs)
# df5=df["国家"]
# for j in range(len(nbs)):
#     print(df5.loc[j])
#     print("的相似国家是")
# # print(nbs[0])
#     for i in nbs[j]:
#         print(df5.loc[i])
#     print("\n")
def similarity(df):
    df["mean"] = df.mean(axis=1)
    other = []
    columns = df.columns
    for x in range(0, df.shape[0]):
        av1 = df.loc[x]["mean"]
        results = []
        for i in range(0, df.shape[0]):
            av = df.loc[i]["mean"]
            q = 0
            q1 = 0
            q2 = 0
            for j in range(0, df.shape[1]):
                m = (df.loc[i][j] - av) * (df.loc[x][j] - av1)
                q = q + m
                m1 = (df.loc[i][j] - av) * (df.loc[i][j] - av)
                q1 = q1 + m1
                m2 = (df.loc[x][j] - av1) * (df.loc[x][j] - av1)
                q2 = q2 + m2

            sim = q / (math.sqrt(q1) * math.sqrt(q2))
            results.append(sim)
        other.append(results)
    return other


##---dimension1
df01 = df_normal[dm1]
# nbs=range(0,91)
result1 = similarity(df01)
print(result1)
result1 = np.array(result1)

colormap = plt.cm.viridis
plt.figure(figsize=(20, 20))
# plt.title('Pearson Correlation of Features', size=15)
ax = sns.heatmap(result1, cmap=colormap, annot=False)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.savefig("result1.png", dpi=400)

# ##dimension2

df02 = df_normal[dm2]
result2 = similarity(df02)
# print(result1)
result2 = np.array(result2)

colormap = plt.cm.viridis
plt.figure(figsize=(20, 20))
# plt.title('Pearson Correlation of Features', size=15)
ax = sns.heatmap(result2, cmap=colormap, annot=False)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.savefig("result2.png", dpi=400)
#

df2 = pd.DataFrame(result1)
df3 = pd.DataFrame(result2)
df_all = pd.concat([df2, df3], axis=1)
df_all.to_excel("df_相似度结果2.xlsx")
