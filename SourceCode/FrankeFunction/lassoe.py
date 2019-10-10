
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
fig = plt.figure()
ax = fig.gca(projection='3d')
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Make data. Why do we have to have 100 to 100?


#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
#Linspace funker ikke, hvorfor?
np.random.seed(2000)
n = 100
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
print(z.shape)
"""
def CreateDesignMatrix_X(x, y, n):
	
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	

	x, y = np.meshgrid(x,y)
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.zeros((10000,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

			X_skl = X[:,1:]
	return X_skl
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
maxdegree = 7
matrixX = CreateDesignMatrix_X(x, y, maxdegree)
print('for degree = ', maxdegree,',shape of matrix is:',matrixX.shape)
"""
# I try making my design matrix
X = np.zeros((n*n, 21))
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
X_skl = X [:,1:]

n1 = 5
lmb = np.linspace(0.01, 0.0001, n1)
print(lmb)
"""clf = Ridge(alpha = lmb)
clf.fit(X_skl, z)
print('beta lasso is',clf.coef_)
"""
beta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(z);
print('beta is', beta)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
#Function for the MSE
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

'''
for i in lmb:
	lasso = Lasso(alpha = i, max_iter = 10e5, tol = 0.0001, fit_intercept = False)
	lasso.fit(X,z)
	betaLasso = lasso.coef_
	#clf = Linear_model.Lasso(alpha = i)
	#clf.fit(X, z)
	# A bit unsure about which design matrix to use. X_skl, or X?
	#betaLasso = list(clf.coef_)
	#betaLasso = np.asarray(betaLasso)
	print('beta lasso is', betaLasso)
	print(betaLasso.shape)
	z_tilde = X@betaLasso
	r2_score1 = r2_score(z, z_tilde)
	print('R2 SCORE IS', r2_score1)
	print('MSE IS',MSE(z, z_tilde) )

'''

lasso1 = Lasso(alpha = 1e-6, max_iter = 10e5, tol = 0.001, fit_intercept = False)
lasso1.fit(X,z)
betaLasso = lasso1.coef_
z_tilde = X@betaLasso

z = FrankeFunction(x, y)
# Plot the surface.

x_plot = x.reshape((n, n))
y_plot = y.reshape((n,n))
z_plot = z_tilde.reshape((n,n))
z1 = z.reshape(n,n)

z_error = (z-z_tilde).reshape((n,n))

surf = ax.plot_surface(x_plot, y_plot, z_plot,  cmap=cm.gray,
                       linewidth=0, antialiased=False)
'''#plot with initial z value
surf = ax.plot_surface(x_plot, y_plot, z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)'''

#plot with z_error to see difference of z and z_tilde
'''
surf = ax.plot_surface(x_plot, y_plot, z_error, cmap=cm.viridis,
                       linewidth=0, antialiased=False) '''
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



