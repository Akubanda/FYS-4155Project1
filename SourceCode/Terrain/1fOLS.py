
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread
from sklearn import preprocessing
from sklearn.linear_model import Lasso
#from sklearn.preprocessing import MinMaxScaler


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data. Why do we have to have 100 to 100?


#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
#Linspace funker ikke, hvorfor?
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

x = np.ravel(x)

y = np.ravel(y)
# Load the terrain
terrain1 = imread('sourceCode/Project1/project1.tif')
z = terrain1
#Normalize the terrain
#scaler = MinMaxScaler(feature_range=(0, 1))
#z = scaler.fit_transform(z)
# summarize transformed data
#np.set_printoptions(precision=3)
z = preprocessing.normalize(z)

n = 100
z = z[:n,:n]
z = np.ravel(z).reshape(-1,1)
print('shape of z is', z.shape)
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

X_skl = X[:,1:]
# Calculating out beta
beta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(z);
#betaRidge = np.linalg.inv(np.transpose(X).dot(X) + 0.0000001*np.identity(21)).dot(np.transpose(X)).dot(z)

'''lasso =Lasso (alpha =0.0001, max_iter = 10e5, tol = 0.0001,fit_intercept=False)
lasso.fit(X_skl,z)
betaLasso = lasso.coef_ '''
print(beta)
#print(beta)
#Kanskje noe galt her?
z_tilde = X.dot(beta)
#z_tilde = X_skl.dot(beta)

print(z_tilde.shape)



#z_tilde = z_tilde.reshape(10,10)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


print('R2 score of the model is' , R2(z,z_tilde))

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

print('MSE for the model is', MSE(z, z_tilde))

var = np.linalg.inv(np.transpose(X).dot(X))*0.01
std = np.sqrt(np.diag(var))
ConfInt = 1.96*std + beta
ConfInter = -1.96*std + beta
print('beta is', beta)
print('Confidence interval plus beta is', ConfInt)
print('Confidence interval minus beta is', ConfInter)


x_plot = x.reshape((n,n))
y_plot = y.reshape((n,n))
z_plot = z_tilde.reshape((n,n))
z1 = z.reshape(n,n)

z_error = (z-z_tilde).reshape((n,n))
surf = ax.plot_surface(x_plot, y_plot, z_plot,  cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# plot with initial z value
#surf = ax.plot_surface(x_plot, y_plot, z1, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

# plot with z_error to see difference of z and z_tilde
'''
surf = ax.plot_surface(x_plot, y_plot, z_error, cmap=cm.gray,
                       linewidth=0, antialiased=False) '''
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
#Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''
from sklearn.model_selection import train_test_split
# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2, random_state=5)
# why are the first two matrices?
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
'''
