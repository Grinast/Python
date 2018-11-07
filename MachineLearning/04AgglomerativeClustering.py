# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:43:03 2018

@author: Cristina
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#Generate random data
#500 points, 5 centres specified, standard deviation of 1.1
#X: Array of shape [n_samples, n_features]. The generated samples.
#y: Array of shape [n_samples]. The integer labels for cluster membership of each sample.
X, y = make_blobs(n_samples = 500, centers = [[4, 4], [-2, -1], [20, 4],[1,1], [0,10]],
                  cluster_std = 1.1)

plt.scatter(X[:,0], X[:,1], marker = ".")

#create agglomerative clustering models
agglom = AgglomerativeClustering(n_clusters = 5, linkage = "complete")
agglom2 = AgglomerativeClustering(n_clusters = 5, linkage = "average")

#fit the modles
agglom.fit(X, y)
agglom2.fit(X, y)

#visualisation
plt.figure(figsize = (12,8))
plt.title("Agglomerative clustering, complete linkage")
# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.
# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
# Get the average distance for X1.
X1 = (X - x_min) / (x_max - x_min)
#This loop displays all data points
for i in range(X1.shape[0]):
       # Replace the data points with their respective cluster value 
       # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
       plt.text(X1[i,0], X1[i,1], str(y[i]), 
                color = plt.cm.nipy_spectral(agglom.labels_[i] / 10.), 
                fontdict = {"weight": "bold", "size": 9})
#remove ticks from both axes
plt.xticks([])
plt.yticks([])
# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()


plt.figure(figsize = (12,8))
plt.title("Agglomerative clustering, average linkage")
for i in range(X1.shape[0]):
       plt.text(X1[i,0], X1[i,1], str(y[i]), 
                color = plt.cm.nipy_spectral(agglom2.labels_[i] / 10.), 
                fontdict = {"weight": "bold", "size": 9})
plt.xticks([])
plt.yticks([])
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()


#Dendogram relative to the agglomerative clustering
#compute the distance matrix
dist_mat = distance_matrix(X,X)
print(dist_mat)

Z = hierarchy.linkage(dist_mat, "complete")
Z2 = hierarchy.linkage(dist_mat, "average")

#compute and show the dendrogram
dendro = hierarchy.dendrogram(Z)
dendro2 = hierarchy.dendrogram(Z2)
