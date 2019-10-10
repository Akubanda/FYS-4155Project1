import numpy as np
import pandas as pd
from IPython.display import display
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing

np.random.seed(2018)

# Load the terrain
terrain1 = imread('DataFiles/project1.tif')
print(terrain1.shape, 'is the shape of the terrain')

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
z = terrain1
#Normalize the terrain
z = preprocessing.normalize(z)
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = z[:n,:n]
z = np.ravel(z).reshape(-1,1)


print('shape of z is', z.shape)
maxdegree = 3
X = CreateDesignMatrix_X(x, y, maxdegree, n)

X_skl = X[:,1:]
# shuffle data

index = np.arange(z.shape[0])
np.random.shuffle(index)
z = z[index]
X_skl = X_skl[index]

#Assign dimensions of matrix X to variables aa and bb
aa, bb = X.shape
print("Shape of X is :", aa,' ', bb)
print("Shape of X skl is :", X_skl.shape)

NumberOfFolds=5
# foldLength is the length of our folds
foldLength = int(aa / NumberOfFolds)
print('fold length,', foldLength)

# splitting our matrix X into K = 5 folds
xFoldShape = (foldLength, bb)
xFolds = np.split(X_skl, NumberOfFolds)
# splitting z (Franke function) into K = 5 folds
yFolds = np.split(z, NumberOfFolds)
print('shape of z is', z.shape)

''' The split command above returns a list.The two functions below convert our
the folds back to an array
X trainset is reshaped to matrix with 0.8 of initial n since our k = 5'''
def xTrainFunction(folds):
    #convert list into an array
    xTrainSet = np.asarray(folds)
    #reshape array to matrix.trainset has 0.8 of n which is 8,000 in this case
    xTrain = xTrainSet.reshape(foldLength*(k-1),bb-1)
    #print('x train set is', xTrain)
    return xTrain

def yTrainFunction(folds):
    #convert list into an arrayn*(k-1)
    yTrainSet = np.asarray(folds)

    #reshape array to a vector of size 0.8 of n which is 8,000
    yTrain = yTrainSet.reshape(foldLength*(k-1),)
    #print('Y train AFTER reshaping is', yTrain)
    return yTrain

# define number of folds as k
k = NumberOfFolds
# declare an array that will store all the MSE scores for each iteration.
mseArray = []
r2ScoreArray = []
interceptArr = []
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

''' This for-loop iterates through the x and y folds. For each iteration, one
fold is selected as test and the rest of the folds as train data'''
for i in range(0,k):
    ''' The general case being used would not apply when i = 4 because using
    index (i+1) for the lists would be out of reference'''
    if (i == (k-1)):
        xTrain = xTrainFunction(xFolds[0:i])
        yTrain = yTrainFunction(yFolds[0:i])
    else:
        xTrain = xTrainFunction(xFolds[:i]+ xFolds[(i+1):])
        yTrain = yTrainFunction(yFolds[:i]+ yFolds[(i+1):])
	#fold i in each iteration is assigned as test set
    xTest = xFolds[i]
    #print('x test  is', xTest)
    yTest = yFolds[i]
    #print('y test 1 is', yTest)
    lasso = Ridge(alpha = 0.0001)
    #lasso = Lasso(alpha = 0.0001, max_iter = 10e5, tol = 0.0001, fit_intercept = False)
    lasso.fit(xTrain,yTrain)

    ypred=lasso.predict(xTest)# mean value
    '''calculate the prediction error of the fitted model when
    predicting the kth part of the data'''

    mse = mean_squared_error(yTest,ypred)
    print('mse = ',mse,' when fold ', (i+1), 'is selected as test data')
    r2Score2 = r2_score(ypred,yTest)

    #print('Variance score:', r2Score )

    print('r2  score:', r2Score2 )
    mseArray.append(mse)
    r2ScoreArray.append(r2Score2)

#print('All the MSEs calculated are:', mseArray)
print('Mean MSE is', np.mean(mseArray))
#print('All the r2 scores calculated are:', r2ScoreArray)
print('Mean r2 score is', np.mean(r2ScoreArray))
