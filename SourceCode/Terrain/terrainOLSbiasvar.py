import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from IPython.display import display
from imageio import imread
from sklearn import preprocessing

np.random.seed(2018)
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
	return X

n_boostraps = 10
z = np.array(terrain1)
z = preprocessing.normalize(z)
# n data points used
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = z[:n,:n]
z = np.ravel(z).reshape(-1,1)
print(z.shape)
maxdegree = 5
X = CreateDesignMatrix_X(x, y, maxdegree, n)
X_skl = X[:,1:]
#empty arrays to store our calculated values
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(X[:,1:], z, test_size=0.2)
# resampling with nbootstraps
for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
    polydegree[degree] = degree
	#calculate error, bias and variance
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
# plot bias variance error for all degrees
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
