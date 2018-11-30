# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:53:35 2018

@author: Cristina
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

"""
Imagine that you are a medical researcher compiling data for a study. You have 
collected data about a set of patients, all of whom suffered from the same 
illness. During their course of treatment, each patient responded to one of 5 
medications, Drug A, Drug B, Drug c, Drug x and y.
Part of your job is to build a model to find out which drug might be appropriate 
for a future patient with the same illness. The feature sets of this dataset are 
Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the 
drug that each patient responded to.
It is a sample of binary classifier, and you can use the training part of the 
dataset to build a decision tree, and then use it to predict the class of a 
unknown patient, or to prescribe it to a new patient.
"""

df = pd.read_csv("drug200.csv")
df.head()

df.shape

#preprocessing
X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].values
X[0:5]
y = df[["Drug"]]

#some features are categorical such as Sex or BP. Sklearn Decision Trees do not 
#handle categorical variables. We can convert these features to numerical values 
#using pandas.get_dummies()
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex = le_sex.fit(["F","M"])
X[:,1] = le_sex.transform(X[:,1])
X[0:5]
#cholesterol
set(X[:,2])
le_chol = preprocessing.LabelEncoder()
le_chol = le_chol.fit(["LOW","NORMAL", "HIGH"])
X[:,2] = le_chol.transform(X[:,2])
X[0:5]
#blood pressure
set(X[:,3])
le_bp = preprocessing.LabelEncoder()
le_bp = le_bp.fit(["NORMAL", "HIGH"])
X[:,3] = le_bp.transform(X[:,3])
X[0:5]

#split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape
y_train.shape

#Modelling
DTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
DTree

DTree.fit(X_train, y_train)

#Prediction
yhat = DTree.predict(X_test)
yhat
print (yhat[0:5])
print (y_test[0:5])

#Evaluation
from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, yhat))

#visualisation
#import matplotlib.pyplot as plt
#from sklearn.externals.six import StringIO
#import pydotplus
#import matplotlib.image as mpimg
#from sklearn import tree
#
#dot_data = StringIO()
#filename = "drugtree.png"
#FeatureNames = df.columns[0:5]
#TargetNames = df["Drug"].unique().tolist()
#out = tree.export_graphviz(DTree, feature_names = FeatureNames, 
#                           out_file = dot_data, class_names = np.unique(y_train),
#                           filled = True, special_characters = True, rotate = False)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100,200))
#plt.imshow(img, interpolation="nearest")

"""
VISUALISATION IS NOT WORKING.
THERE IS AN ISSUE WITH GRAPHVIZ. I DO NOT WANT TO ADD IT TO PATH. WORKAROUND?
"""