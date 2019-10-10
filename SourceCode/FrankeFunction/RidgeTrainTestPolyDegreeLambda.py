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
np.random.seed(5000)
n = 10
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(n,n)
z = FrankeFunction(x, y)

z = np.ravel(z)
z = z.reshape(-1, 1)

def CreateDesignMatrix_X(x, y, n):
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
	X = np.zeros((100,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

			X_skl = X[:,1:]
	return X_skl
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
maxdegree = 30
matrixX = CreateDesignMatrix_X(x, y, maxdegree)
print('for degree = ', maxdegree,',shape of matrix is:',matrixX.shape)
nlambdas = 3
lambdas = np.logspace(-6, 1, nlambdas)
#X[:,1:]
X_train, X_test, y_train, y_test = train_test_split(matrixX, z, test_size=0.5)
trainerrorDegree = np.zeros(maxdegree)
testerrorDegree  = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
testerror = np.empty((maxdegree, nlambdas))
trainerror = np.empty((maxdegree, nlambdas))
for degree in range(1, maxdegree+1):

    NumberOfColumns = int(((degree+1)*(degree+2)/2)-1)
    print(NumberOfColumns)
    l = 0
    # for each degree, fit model into train set with correct number of beta values

    for lmb in lambdas:
        print('lmb is', lmb)

        betaRidge = np.linalg.inv(np.transpose(X_train[:,:NumberOfColumns]).dot(X_train[:,:NumberOfColumns]) + lmb*np.identity(NumberOfColumns)).dot(np.transpose(X_train[:,:NumberOfColumns])).dot(y_train)
        ytilde = X_test[:,:NumberOfColumns]@betaRidge
        ypred = X_train[:,:NumberOfColumns]@betaRidge
        testerror[degree-1,l] = mean_squared_error(y_test, ytilde)
        trainerror[degree-1,l] = mean_squared_error(y_train, ypred)
        l = l+ 1
        #testerrorDegree[degree] = testerror
        #trainerrorDegree[degree] = trainerror
        polydegree[degree-1] = degree
# print each lambda value for each train and test error

for i, lmd in enumerate(lambdas):
    print(testerror[:, i].shape)
    print(testerror[i, :].shape)
    plt.plot(polydegree, testerror[:, i], label='test error for lambda {}'.format(lambdas[i]))
    plt.plot(polydegree, trainerror[:, i], label= 'train error for lambda {}'.format(lambdas[i]))

plt.xlabel('Polynomial degree')
plt.ylabel('[MSE]')
plt.legend()
plt.show()
