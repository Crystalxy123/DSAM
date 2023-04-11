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

df_normal.to_excel("normalized.xlsx")