import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
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
maxdegree = 20
X = CreateDesignMatrix_X(x, y, maxdegree,n)

X_skl = X[:,1:]
nlambdas = 5
lambdas = np.linspace(0, 1, nlambdas)
#X[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X_skl, z, test_size=0.2)
trainerrorDegree = np.zeros(maxdegree)
testerrorDegree  = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
testerror = np.empty((maxdegree, nlambdas))
trainerror = np.empty((maxdegree, nlambdas))
for degree in range(1, maxdegree+1):

    NumberOfColumns = int(((degree+1)*(degree+2)/2)-1)
    l = 0
    for lmb in lambdas:
		lasso = Lasso(alpha = lmb, max_iter = 10e5, tol = 0.0001, fit_intercept = False)
		lasso.fit(X_train[:,:NumberOfColumns],y_train)
		betaLasso = lasso.coef_
		ytilde = X_test[:,:NumberOfColumns]@betaLasso
		print(ytilde.shape)
		print(y_test.shape)
		ypred = X_train[:,:NumberOfColumns]@betaLasso
		testerror[degree-1,l] = mean_squared_error(y_test, ytilde)
		trainerror[degree-1,l] = mean_squared_error(y_train, ypred)
		l = l+ 1
		#testerrorDegree[degree] = testerror
		#trainerrorDegree[degree] = trainerror
		polydegree[degree-1] = degree
for i, lmd in enumerate(lambdas):
    print(testerror[:, i].shape)
    print(testerror[i, :].shape)
    plt.plot(polydegree, testerror[:, i], label='test error for lambda {}'.format(lambdas[i]))
    plt.plot(polydegree, trainerror[:, i], label= 'train error for lambda {}'.format(lambdas[i]))

plt.xlabel('Polynomial degree')
plt.ylabel('log10[MSE]')
plt.legend()
plt.show()
