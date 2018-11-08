# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:35:59 2018

@author: Cristina
"""

"""
Imagine that an automobile manufacturer has developed prototypes for a new vehicle.
Before introducing the new model into its range, the manufacturer wants to determine 
which existing vehicles on the market are most like the prototypes--that is, how 
vehicles can be grouped, which group is the most similar with the model, and 
therefore which models they will be competing against.
Our objective here, is to use clustering methods, to find the most distinctive 
clusters of vehicles. It will summarize the existing vehicles and help 
manufacturers to make decision about the supply of new models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#import data
df = pd.read_csv("cars_clus.csv")
print ("Shape of dataset: ", df.shape)
df.head()

#data cleaning (preprocessing)
print ("Shape of dataset before cleaning: ", df.size)
df[["sales", "resale", "type", "price", "engine_s", "horsepow", "wheelbas", 
    "width", "length", "curb_wgt", "fuel_cap", "mpg", "lnsales"]] = df[["sales",
    "resale", "type", "price", "engine_s", "horsepow", "wheelbas", "width", 
    "length", "curb_wgt", "fuel_cap", "mpg", "lnsales"]].apply(pd.to_numeric, 
    errors="coerce")
df = df.dropna()
df = df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", df.size)

#features selection         
featureset = df[["engine_s",  "horsepow", "wheelbas", "width", "length", 
                  "curb_wgt", "fuel_cap", "mpg"]]
featureset.head()

#Normaliation to (0,1)
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mat = min_max_scaler.fit_transform(x)
feature_mat


#CLUSTERING USING SCIPY
import scipy
import pylab
from scipy.cluster.hierarchy import linkage, dendrogram

#calculate distance matrix
leng = feature_mat.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mat[i], feature_mat[j])

#compute hierarchy
#Possible options for the linkage are: single, complete, average, weighted, centroid
Z = linkage(D, "complete")

#plot the dendrogram
fig = pylab.figure(figsize = (10,24))
def llf(id):
    return "[%s %s %s]" % (df["manufact"][id], df["model"][id], 
            int(float(df["type"][id])) )

dendro = dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, 
                    orientation = "right")


#CLUSTERING USING SCIKIT-LEARN
from scipy.spatial import distance_matrix 
dist_mat = distance_matrix(feature_mat, feature_mat)

agglom = AgglomerativeClustering(n_clusters = 6, linkage = "complete")
agglom.fit(feature_mat)
agglom.labels_

df["cluster_"] = agglom.labels_
df.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

#Visualisation
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i],str(subset["model"][i]), 
                 rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, 
                label="cluster"+str(label),alpha=0.5)
plt.legend()
plt.title("Clusters")
plt.xlabel("horsepow")
plt.ylabel("mpg")
plt.show()

#here are 2 types of vehicles in our dataset, "truck" (value of 1 in the type 
# column) and "car" (value of 1 in the type column)
df.groupby(["cluster_","type"])["cluster_"].count()

#Now we can look at the characteristics of each cluster:
agg_cars = df.groupby(["cluster_","type"])["horsepow","engine_s","mpg","price"].mean()
agg_cars

#notice that we did not use type , and price of cars in the clustering process, 
#but Hierarchical clustering could forge the clusters and discriminate them 
#with quite high accuracy.

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], "type="+str(int(i)) + 
                 ", price="+str(int(subset.loc[i][3]))+"k")
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, 
                label = "cluster"+str(label))
plt.legend()
plt.title("Clusters")
plt.xlabel("horsepow")
plt.ylabel("mpg")
plt.show()