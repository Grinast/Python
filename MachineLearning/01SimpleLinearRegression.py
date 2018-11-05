# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:29:39 2018

@author: Cristina
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#read dataset
df = pd.read_csv("FuelConsumptionCo2.csv")

df.head()

#explore data
df.describe()

#select some features
cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
cdf.head()

#graphic visualisation
cdf.hist()
plt.show()

#plot engine size aginst emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="red")
plt.xlabel("Engine size")
plt.ylabel("C02 emissions")
plt.show()

#plot fuel consumption aginst emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="orange")
plt.xlabel("Fuel consumption_combined")
plt.ylabel("C02 emissions")
plt.show()

#plot cylinders aginst emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="purple")
plt.xlabel("Cylinders")
plt.ylabel("C02 emissions")
plt.show()

#split dataset into training (80%) and testing (20%)
mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

#SIMPLE REGRESSION MODELS
from sklearn import linear_model
from sklearn.metrics import r2_score

def SimpleLinRegr(train_x, train_y, xlabel="Independent variable", ylabel="Dependent variable"):
    """
    train_x: training values of independent variable (array)
    train_y: training values of dependent variable (array)
    prints simple linear regression model, its coefficients and its accuracy
    returns regression model
    """
    #create model
    regr = linear_model.LinearRegression()
    #fit model
    regr.fit(train_x, train_y)
    
    #plot output
    plt.scatter(train_x, train_y, color="red")
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r", color = "blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()    
    
    #coefficients
    print("Coefficients: ", regr.coef_)
    print("Intercept: ", regr.intercept_)
    
    return regr

    
def EvalSimpleLinRegr(test_x, test_y, regr):
    test_y_ = regr.predict(test_x)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Mean square error: %.2f" % np.mean((test_y_ - test_y)**2))
    print("R2 score: %.2f" % r2_score(test_y_ , test_y))

    
#Engine size
regr_eng = SimpleLinRegr(np.asanyarray(train[["ENGINESIZE"]]), 
                      np.asanyarray(train[["CO2EMISSIONS"]]), 
                      "Engine size", "C02 emissions")

EvalSimpleLinRegr(np.asanyarray(test[["ENGINESIZE"]]), 
                  np.asanyarray(test[["CO2EMISSIONS"]]), regr_eng)

#Fuel consumption
regr_fuel = SimpleLinRegr(np.asanyarray(train[["FUELCONSUMPTION_COMB"]]),
                          np.asanyarray(train[["CO2EMISSIONS"]]),
                          "Fuel consumption_combined", "C02 emissions")
EvalSimpleLinRegr(np.asanyarray(test[["FUELCONSUMPTION_COMB"]]), 
                  np.asanyarray(test[["CO2EMISSIONS"]]), regr_fuel)

#Cylinders
regr_cyl = SimpleLinRegr(np.asanyarray(train[["CYLINDERS"]]), 
                         np.asanyarray(train[["CO2EMISSIONS"]]),
                         "Cylinders", "C02 emissions")

EvalSimpleLinRegr(np.asanyarray(test[["CYLINDERS"]]), 
                  np.asanyarray(test[["CO2EMISSIONS"]]), regr_cyl)


#Multiple linear regression
print("Multiple linear regression")
X = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]]
train_X = X[mask]
train_y = df.CO2EMISSIONS[mask]
test_X = X[~mask]
test_y = df.CO2EMISSIONS[~mask]
mult_regr = linear_model.LinearRegression()
mult_regr.fit(train_X,train_y)
print("Coefficients: ", mult_regr.coef_)
print("Intercept: ", mult_regr.intercept_)
#evaluation
test_y_ = mult_regr.predict(test_X)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean square error: %.2f" % np.mean((test_y_ - test_y)**2))
print("R2 score: %.2f" % r2_score(test_y_ , test_y))