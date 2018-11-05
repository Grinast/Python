# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:08:20 2018

@author: Cristina
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#K-Means on randomly generated dataset

#generate random dataset
np.random.seed(0)
#5000 points, 4 centres specified, standard deviation of 0.9
#X: Array of shape [n_samples, n_features]. The generated samples.
#y: Array of shape [n_samples]. The integer labels for cluster membership of each sample.
X, y = make_blobs(n_samples = 5000, centers = [[4, 4], [-2, -1], [2, -3],[1,1]],
                  cluster_std = 0.9)

plt.scatter(X[:,0], X[:,1], marker = ".")


def k_means(X, clusters):
       """
       X: Array of shape [n_samples, 2].
       clusters: number of desired clusters/centroids
       Returns the fitted KMeans model and plots the result.
       """
      
       #set up k-means
       
       #n_clusters: The number of clusters to form as well as the number of centroids to generate.
       #init: Initialization method of the centroids. k-means++: Selects initial cluster
       # centers for k-mean clustering in a smart way to speed up convergence.
       #n_init: Number of times the k-means algorithm will be run with different centroid seeds.
       k_means = KMeans(clusters, "k-means++", 12)
       #fit model
       k_means.fit(X)
       #grab the labels of each point in the model
       k_means_labels = k_means.labels_
       #grab the coordinates of the centroids in the model
       k_means_centers = k_means.cluster_centers_
       
       
       #visualise results
       
       fig = plt.figure(figsize=(12,8))
       #Colors uses a color map, which will produce an array of colors based on
       # the number of labels there are.
       colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))
       # Create a plot
       ax = fig.add_subplot(1, 1, 1)
       #For loop that plots the data points and centroids.
       for k, col in zip(range(clusters), colors):
              #Create a list of all data points in the given cluster. Points in the 
              # cluster are labeled as true, else they are labeled as false.
              my_members = (k_means_labels == k)
              #Define the centroid
              centroid = k_means_centers[k]
              #Plot the datapoints with color col
              ax.plot(X[my_members, 0], X[my_members, 1], "w", 
                      markerfacecolor = col, marker = ".")
              # Plots the centroids with specified color, but with a darker outline
              ax.plot(centroid[0], centroid[1], "o", color = col, 
                      markerfacecolor = col, markersize = 6)

k_means(X, 4)
k_means(X, 3)
k_means(X, 5)