import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from random import random, seed
from sklearn.utils import resample
from sklearn import preprocessing
from imageio import imread

# Load the terrain
terrain1 = imread('DataFiles/project1.tif')
print(terrain1.shape, 'is the shape of the terrain')

def CreateDesignMatrix_X(x, y, n, num):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""

	x, y = np.meshgrid(x,y)
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.zeros((num * num,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

			X
	return X
z = terrain1
#Normalize the terrain
z = preprocessing.normalize(z, norm='l1')
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = z[:n,:n]
z = np.ravel(z).reshape(-1,1)

print('shape of z is', z.shape)
maxdegree = 50
X = CreateDesignMatrix_X(x, y, maxdegree,n)

X_skl = X[:,1:]
nlambdas = 5
lambdas = np.linspace(0, 1, nlambdas)
#X[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
trainerrorDegree = np.zeros(maxdegree)
testerrorDegree  = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
testerror = np.empty((maxdegree, nlambdas))
trainerror = np.empty((maxdegree, nlambdas))
for degree in range(1, maxdegree+1):

    NumberOfColumns = int(((degree+1)*(degree+2)/2)-1)

    linreg = LinearRegression()
    linreg.fit(X_train[:,:NumberOfColumns],y_train)
    beta = linreg.coef_
    ytilde = linreg.predict(X_train[:,:NumberOfColumns])
    ypred = linreg.predict(X_test[:,:NumberOfColumns])
    #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
	#betaRidge = np.linalg.inv(np.transpose(X_train[:,:NumberOfColumns]).dot(X_train[:,:NumberOfColumns]) + lmb*np.identity(NumberOfColumns)).dot(np.transpose(X_train[:,:NumberOfColumns])).dot(y_train)
    #ytilde = X_train[:,:NumberOfColumns] @ beta
    #ypred = X_test[:,:NumberOfColumns] @ beta
    trainerrorDegree[degree-1] = mean_squared_error(y_train, ytilde)
    testerrorDegree[degree-1] = mean_squared_error(y_test, ypred)
    polydegree[degree-1] = degree

'''
for i, lmd in enumerate(lambdas):
    print(testerror[:, i].shape)
    print(testerror[i, :].shape)
    plt.plot(polydegree, testerror[:, i], label='test error for {}'.format(lambdas[i]))
    plt.plot(polydegree, trainerror[:, i], label= 'train error for lambda {}'.format(lambdas[i]))'''
plt.plot(polydegree, trainerrorDegree)
plt.plot(polydegree, testerrorDegree)

plt.xlabel('Polynomial degree')
plt.ylabel('log10[MSE]')
plt.legend()
plt.show()
