# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:55:03 2018

@author: Cristina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

"""
The target field, custcat, has four possible values that correspond to the four 
customer groups, as follows: 1-Basic Service 2-E-Service 3-Plus Service 4-Total 
Service.
Our objective is to build a classifier to predict the class of unknown cases. 
"""
df = pd.read_csv("teleCust1000t.csv")
df.head()

#data analysis
#how many of each class is in the dataset?
df["custcat"].value_counts()
df.hist(column = "income", bins = 50)
df.hist(column = "gender")
df.hist(column = "age")

#Convert dataframe (last column separetely) into numpy array
df.columns
X = df[["region", "tenure", "age", "marital", "address", "income", "ed", 
         "employ", "retire", "gender", "reside"]].values
X        
y = df["custcat"].values
y

#Data Standardization give data zero mean and unit variance, it is good practice, 
#especially for algorithms such as KNN which is based on distance of cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X

#split into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("Train set: ", X_train.shape, y_train.shape) 
print("Testn set: ", X_test.shape, y_test.shape)

#CLASSIFICATION
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def k_neigh_test(neigh, X_train, y_train, X_test, y_test):
    """
    Computes the accuracy of a given k-nearest neighbours model given the model 
    itself (neigh), and split data
    """
    #predicting
    yhat = neigh.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, neigh.predict(X_train))
    acc_test = metrics.accuracy_score(y_test, yhat)

    #accuracy evaluation
    print("Accuracy on training set: ", acc_train)
    print("Accuracy on test set: ", acc_test)
    return acc_train, acc_test

#training
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
k_neigh_test(neigh, X_train, y_train, X_test, y_test)
k = 5
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
k_neigh_test(neigh, X_train, y_train, X_test, y_test)
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
k_neigh_test(neigh, X_train, y_train, X_test, y_test)

#find best value for k
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfusionMat = []
for k in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[k-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[k-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])
mean_acc
std_acc
    
#plot accuracy as a function of k
plt.figure()
plt.plot(range(1,Ks),mean_acc,"g")
plt.fill_between(range(1,Ks), mean_acc - 1*std_acc,mean_acc + 1*std_acc, alpha=0.10)
plt.legend(("Accuracy ", "+/- 3xstd"))
plt.ylabel("Accuracy ")
plt.xlabel("Number of Neighbors (k)")
plt.tight_layout()
plt.show()

print( "Best accuracy: ", mean_acc.max(), "when k=", mean_acc.argmax()+1) 