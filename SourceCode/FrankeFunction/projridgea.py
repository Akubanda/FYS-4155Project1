
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
fig = plt.figure()
ax = fig.gca(projection='3d')
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
import statistics
import scipy
from sklearn.metrics import r2_score
# Make data. Why do we have to have 100 to 100?
np.random.seed(50000)

#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
#Linspace funker ikke, hvorfor?
n = 40
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 
noise = 0.1*np.random.randn(n,n)
z = FrankeFunction(x, y) + noise

x = np.ravel(x)

y = np.ravel(y)
z = np.ravel(z)
# I try making my design matrix

X = np.zeros((n*n,21))
X[:,0] = 1
X[:,1] = x
X[:,2] = y
X[:,3] = y*y
X[:,4] = x*x
X[:,5] = x*y
X[:,6] = x*x*x
X[:,7] = y*y*y
X[:,8] = y*x*x
X[:,9] = x*y*y
X[:,10] = x*x*x*x
X[:,11] = y*y*y*y
X[:,12] = x*x*y*y
X[:,13] = x*y*y*y
X[:,14] = x*x*x*y
X[:,15] = x*x*x*x*x
X[:,16] = y*y*y*y*y
X[:,17] = x*x*x*x*y
X[:,18] = x*y*y*y*y
X[:,19] = x*x*x*y*y
X[:,20] = x*x*y*y*y
'''
X[:,21] = x*x*x*x*x*x
X[:,22] = x*x*x*x*x*y
X[:,23] = x*x*x*x*y*y
X[:,24] = x*x*x*y*y*y
X[:,25] = x*x*y*y*y*y
X[:,26] = x*y*y*y*y*y
X[:,27] = y*y*y*y*y*y
X[:,28] = x*x*x*x*x*x*x
X[:,29] = x*x*x*x*x*x*y
X[:,30] = x*x*x*x*x*y*y
X[:,31] = x*x*x*x*y*y*y
X[:,32] = x*x*x*y*y*y*y
X[:,33] = x*x*y*y*y*y*y
X[:,34] = x*y*y*y*y*y*y
X[:,35] = y*y*y*y*y*y*y
X[:,36] = y*y*y*y*y*y*y*x
X[:,37] = y*y*y*y*y*y*x*x
X[:,38] = y*y*y*y*y*x*x*x
X[:,39] = y*y*y*y*x*x*x*x
X[:,40] = y*y*y*x*x*x*x*x
X[:,41] = y*y*x*x*x*x*x*x
X[:,42] = y*x*x*x*x*x*x*x
X[:,43] = x*x*x*x*x*x*x*x
X[:,44] = y*y*y*y*y*y*y*y
X[:,45] = y*y*y*y*y*y*y*y*x
X[:,46] = y*y*y*y*y*y*y*x*x
X[:,47] = y*y*y*y*y*y*x*x*x
X[:,48] = y*y*y*y*y*x*x*x*x
X[:,49] = y*y*y*y*x*x*x*x*x
X[:,50] = y*y*y*x*x*x*x*x*x
X[:,51] = y*y*x*x*x*x*x*x*x
X[:,52] = y*x*x*x*x*x*x*x*x
X[:,53] = y*y*y*y*y*y*y*y*y
X[:,54] = x*x*x*x*x*x*x*x*x
X[:,55] = x*x*x*x*x*x*x*x*x*y
X[:,56] = x*x*x*x*x*x*x*x*y*y
X[:,57] = x*x*x*x*x*x*x*y*y*y
X[:,58] = x*x*x*x*x*x*y*y*y*y
X[:,59] = x*x*x*x*x*y*y*y*y*y
X[:,60] = x*x*x*x*y*y*y*y*y*y
X[:,61] = x*x*x*y*y*y*y*y*y*y
X[:,62] = x*x*y*y*y*y*y*y*y*y
X[:,63] = x*y*y*y*y*y*y*y*y*y
X[:,64] = y*y*y*y*y*y*y*y*y*y
X[:,65] = x*x*x*x*x*x*x*x*x*x
X[:,66] = x*x*x*x*x*x*x*x*x*x*y
X[:,67] = x*x*x*x*x*x*x*x*x*y*y
X[:,68] = x*x*x*x*x*x*x*x*y*y*y
X[:,69] = x*x*x*x*x*x*x*y*y*y*y
X[:,70] = x*x*x*x*x*x*y*y*y*y*y
X[:,71] = x*x*x*x*x*y*y*y*y*y*y
X[:,72] = x*x*x*x*y*y*y*y*y*y*y
X[:,73] = x*x*x*y*y*y*y*y*y*y*y
X[:,74] = x*x*y*y*y*y*y*y*y*y*y
X[:,75] = x*y*y*y*y*y*y*y*y*y*y
X[:,76] = x*x*x*x*x*x*x*x*x*x*x
X[:,77] = y*y*y*y*y*y*y*y*y*y*y
'''
#Define my Lambdas

n1 = 2
lambdas = np.linspace(1e-6, 10, n1)


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
	var = np.diag(0.01*((np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21))).dot(np.transpose(X).dot(X)).dot(np.transpose(np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21))))))
	print('Var' , var)
	std = (np.sqrt(var))
	print(std.shape)
	betaRidge = np.linalg.inv(np.transpose(X).dot(X) + lmb*np.identity(21)).dot(np.transpose(X)).dot(z)
	ConfInt = 1.96*std + betaRidge
	ConfInter = -1.96*std + betaRidge
	print('BetaRidge is', betaRidge)
	print('The Confidence interval plus  is', ConfInt)
	print('The Confidence interval minus is', ConfInter)
	z_tilde = X@betaRidge
	print('R2 score', r2_score(z_tilde, z))
	#print('R2 SCORE IS', R2(z,z_tilde))
	print('MSE IS',MSE(z, z_tilde) )	
betaRidge = np.linalg.inv(np.transpose(X).dot(X) + 1e-6*np.identity(21)).dot(np.transpose(X)).dot(z)

z_tilde = X@betaRidge

z = FrankeFunction(x, y)
# Plot the surface.

x_plot = x.reshape((n, n))
y_plot = y.reshape((n,n))
z_plot = z_tilde.reshape((n,n))
z1 = z.reshape(n,n)

z_error = (z-z_tilde).reshape((n,n))
'''
surf = ax.plot_surface(x_plot, y_plot, z_plot,  cmap=cm.gray,
                       linewidth=0, antialiased=False)'''
'''#plot with initial z value
surf = ax.plot_surface(x_plot, y_plot, z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)'''

#plot with z_error to see difference of z and z_tilde

surf = ax.plot_surface(x_plot, y_plot, z_error, cmap=cm.viridis,
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




from sklearn.model_selection import train_test_split
# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2, random_state=5)
# why are the first two matrices?


