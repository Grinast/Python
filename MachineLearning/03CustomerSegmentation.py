# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:01:52 2018

@author: Cristina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#import data
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


#Pre-processing

#drop address column
df = cust_df.drop("Address", axis=1)
df.head()
#normalise data
X = df.values[:,1:]
X = np.nan_to_num(X)
std_df = StandardScaler().fit_transform(X)
std_df


#Modeling (using K-means clusters)
clusterNum = 3
k_means = KMeans(clusterNum, "k-means++", 12)
k_means.fit(X)
labels = k_means.labels_
print("Labels: ", labels)


#Insight
df["Cluster"] = labels
df.head()
#Check the centroid values by averaging the features in each cluster.
df.groupby("Cluster").mean()
#look at the distribution of customers based on their age and income
#area is proportional to Education (X[:,1])
area = np.pi * X[:,1] ** 2
#X[:,0] is the Age, X[:,3] is the Income
plt.scatter(X[:,0], X[:,3], s = area, c = labels.astype(np.float), alpha = 0.5)
plt.xlabel("Age", fontsize=18)
plt.ylabel("Income", fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize = (8,6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel("Education")
ax.set_ylabel("Age")
ax.set_zlabel("Income")
ax.scatter(X[:,1], X[:,0], X[:,3], c = labels.astype(np.float))