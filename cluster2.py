import matplotlib
import pandas as pd
#K-means clustering
# Import the requried packages
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
matplotlib.use('TkAgg')
import warnings

warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



df=pd.read_excel("4.xlsx")
print(df.columns)

cls=df
# Convert dataframe into numpy arrays
X =cls.values
# Scaling the data so that all the features/attributes become comparable
# scaler = StandardScaler()
X_scaled = X
# print(X_scaled)
# Task 3: Build Model: using Elbow Method to find the optimal K
sse = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, n_init=20, random_state=1)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 8), sse, 'b-*')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('K_means4.png',dpi=400)
plt.show()

# Build the model using the optimal n_cluster
kmeans = KMeans(n_clusters=4, n_init=20, random_state=1)
# Fit the model and assign each data point to the cluster
y_pred = kmeans.fit_predict(X_scaled) # fit and then predict
#  Visualise the Clusters
plt.scatter(X_scaled[:,0], X_scaled[:,1],s=800,c=y_pred)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.xlabel('Similarity of National Base')
plt.ylabel('Similarity of Mass Base')
plt.savefig('K_means2.png', dpi=400)

print(X_scaled[:,0])
print(X_scaled[:,1])
print(y_pred)
print(len(y_pred))