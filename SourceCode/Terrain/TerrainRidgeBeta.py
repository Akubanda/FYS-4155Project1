from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from random import random, seed
from sklearn.utils import resample
import statistics
import scipy
from imageio import imread
from sklearn import preprocessing

# Load the terrain
terrain1 = imread('DataFiles/project1.tif')
z = terrain1
#Normalize the terrain
z = preprocessing.normalize(z, norm='l1')
#np.random.seed(2018)

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = z[:n,:n]
z = np.ravel(z).reshape(-1,1)
# create design matrix
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
	X = np.ones((num*num,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X
maxdegree = 5
X = CreateDesignMatrix_X(x, y, maxdegree, n)

X_skl = X[:,1:]
'''# shuffle data
index = np.arange(z.shape[0])
np.random.shuffle(index)
z = z[index]
X_skl = X_skl[index]
X = X[index] '''
n1 = 10
lambdas = np.linspace(0, 1, n1)

# Function for the R2 score
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
#Function for the MSE
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# Below is the code used to do calculation for ridge.
for lmb in lambdas:
# Calculating out beta
	var = 0.01*((np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21))).dot(np.transpose(X).dot(X)).dot(np.transpose(np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21)))))
	std = np.sqrt(np.diag(var))
	betaRidge = np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21)).dot(np.transpose(X)).dot(z)
	ConfInt = 1.96*std + betaRidge
	ConfInter = -1.96*std + betaRidge
	print('BetaRidge is', betaRidge)
	print('The Confidence interval plus  is', ConfInt)
	print('The Confidence interval minus is', ConfInter)
	z_tilde = X@betaRidge
	print('R2 SCORE IS', R2(z,z_tilde))
	print('MSE IS',MSE(z, z_tilde) )
betaRidge = np.linalg.inv(np.transpose(X).dot(X) + 0.0000001*np.identity(21)).dot(np.transpose(X)).dot(z)

z_tilde = X@betaRidge

x_plot = x.reshape((n, n))
y_plot = y.reshape((n, n))
z_plot = z_tilde.reshape((n,n))
z1 = z.reshape(n,n)

z_error = (z-z_tilde).reshape((n,n))
surf = ax.plot_surface(x_plot, y_plot, z_plot,  cmap=cm.gray,
                       linewidth=0, antialiased=False)
#plot with initial z value
surf = ax.plot_surface(x_plot, y_plot, z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#plot with z_error to see difference of z and z_tilde

surf = ax.plot_surface(x_plot, y_plot, z_error, cmap=cm.gray,
                       linewidth=0, antialiased=False)
#Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
#Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
