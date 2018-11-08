# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:48:23 2018

@author: Cristina
"""
"""
Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#np.random.seed(0)

#Generate random data
def CreateDataPoints(CentroidLoc, NumSamples, ClusterDeviation):
    X, y = make_blobs(n_samples = NumSamples, centers = CentroidLoc, 
                      cluster_std = ClusterDeviation)
    #Standardise features by setting mean = 0 and variance = 1
    X = StandardScaler().fit_transform(X)
    return X, y


X, y = CreateDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.6)
    

#Modelling
#Epsilon: if enought points are within a circle of radius epsilon from a given 
# point P, then P is a Core Point (dense area)
#MinSamples: the minimum number of data points we want in a neighborhood to 
# define a cluster.
epsilon = 0.3
MinSamples = 7
db = DBSCAN(eps = epsilon, min_samples = MinSamples).fit(X)
labels = db.labels_
unique_labels = set(labels)
unique_labels

#Distinguish outliers
#True: data points in clusters. False: outliers.
#First, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
n_clusters


#Visualisation
# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors
#plot data points
for k, col in zip(unique_labels, colors):
    #noise
    if k == -1: 
        col = "k"
    class_member_mask = (labels == k)
    
    #plot clustered points
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:,1], c = col, marker = u"o", alpha = 0.5)
    #plot outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:,1], c = col, marker = u"o", alpha = 0.5)
plt.show()


#Repeat classification using K-Means clustering
from sklearn.cluster import KMeans

k = 3
k_means = KMeans(k, "k-means++", 12).fit(X)
#grab the labels of each point in the model
k_means_labels = k_means.labels_
#grab the coordinates of the centroids in the model
k_means_centers = k_means.cluster_centers_
   
#visualise results
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.show()
    