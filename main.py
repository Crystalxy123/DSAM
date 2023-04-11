# Import the requried packages
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model, neighbors, tree, svm, ensemble, neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate

import warnings

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')
#---数据预处理
df = pd.read_excel("data_DSAM.XLSX")
df1 = df[["Infection","Government risk management efficiency","Emergency preparedness","Quality and Accessibility of Care Index","education level",
          "percentage of population of low age","Population density","Mass living level","Monitoring and diagnosis"]]

#---数据可视化
#distribution
fig, ax =plt.subplots(3,3,constrained_layout=True, figsize=(16, 9))
axesSub = sns.boxplot(data=df1['Infection'], ax=ax[0][0], width=0.3)
axesSub.set_title('Infection')
axesSub = sns.boxplot(data=df1['Government risk management efficiency'], ax=ax[0][1], width=0.3)
axesSub.set_title('Government risk management efficiency')
axesSub = sns.boxplot(data=df1['Emergency preparedness'], ax=ax[0][2], width=0.3)
axesSub.set_title('Emergency preparedness')
axesSub = sns.boxplot(data=df1['Quality and Accessibility of Care Index'], ax=ax[1][0], width=0.3)
axesSub.set_title('Quality and Accessibility of Care Index')
axesSub = sns.boxplot(data=df1['education level'], ax=ax[1][1], width=0.3)
axesSub.set_title('education level')
axesSub = sns.boxplot(data=df1['percentage of population of low age'], ax=ax[1][2], width=0.3)
_ = axesSub.set_title('percentage of population of low age')
axesSub = sns.boxplot(data=df1['Population density'], ax=ax[2][0], width=0.3)
axesSub.set_title('Population density')
axesSub = sns.boxplot(data=df1['Mass living level'], ax=ax[2][1], width=0.3)
axesSub.set_title('Mass living level')
axesSub = sns.boxplot(data=df1['Monitoring and diagnosis'], ax=ax[2][2], width=0.3)
axesSub.set_title('Monitoring and diagnosis')
plt.savefig("box_distribution.png",dpi=400)
plt.show()
#Person
colormap = plt.cm.viridis
plt.figure(figsize=(50,50))
plt.title('Pearson Correlation of Features', size=20)
ax = sns.heatmap(df1.corr(), cmap=colormap, annot=True,annot_kws={"fontsize":5})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.savefig("Person.png" , dpi = 400)
#
# #infection
# g = sns.PairGrid(data=df1, hue= 'Infection', height=2.5)
# g = g.map(sns.scatterplot)
# g = g.add_legend()
# plt.show()
# plt.savefig("Infection correlation.png",dpi=400)
#
