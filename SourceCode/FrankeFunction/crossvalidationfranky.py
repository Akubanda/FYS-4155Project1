from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
np.random.seed(5000)

#Linspace funker ikke, hvorfor?
n=40
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y) + 0.1*np.random.randn(n, n)

x = np.ravel(x)

y = np.ravel(y)
z = np.ravel(z)
print(z.shape)
# I try making my design matrix

X = np.zeros((n*n, 36))

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
'''
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
X[:,78] = x*x*x*x*x*x*x*x*x*x*x*y
X[:,79] = x*x*x*x*x*x*x*x*x*x*y*y
X[:,80] = x*x*x*x*x*x*x*x*x*y*y*y
X[:,81] = x*x*x*x*x*x*x*x*y*y*y*y
X[:,82] = x*x*x*x*x*x*x*y*y*y*y*y
X[:,83] = x*x*x*x*x*x*y*y*y*y*y*y
X[:,84] = x*x*x*x*x*y*y*y*y*y*y*y
X[:,85] = x*x*x*x*y*y*y*y*y*y*y*y
X[:,86] = x*x*x*y*y*y*y*y*y*y*y*y
X[:,87] = x*x*y*y*y*y*y*y*y*y*y*y
X[:,88] = x*y*y*y*y*y*y*y*y*y*y*y
X[:,89] = x*x*x*x*x*x*x*x*x*x*x*x
X[:,90] = y*y*y*y*y*y*y*y*y*y*y*y
X[:,91] = x*x*x*x*x*x*x*x*x*x*x*x*y
X[:,92] = x*x*x*x*x*x*x*x*x*x*x*y*y
X[:,93] = x*x*x*x*x*x*x*x*x*x*y*y*y
X[:,94] = x*x*x*x*x*x*x*x*x*y*y*y*y
X[:,95] = x*x*x*x*x*x*x*x*y*y*y*y*y
X[:,96] = x*x*x*x*x*x*x*y*y*y*y*y*y
X[:,97] = x*x*x*x*x*x*y*y*y*y*y*y*y
X[:,98] = x*x*x*x*x*y*y*y*y*y*y*y*y
X[:,99] = x*x*x*x*y*y*y*y*y*y*y*y*y
X[:,100] =x*x*x*y*y*y*y*y*y*y*y*y*y
X[:,101] =x*x*y*y*y*y*y*y*y*y*y*y*y
X[:,102] =x*y*y*y*y*y*y*y*y*y*y*y*y
X[:,103] =x*x*x*x*x*x*x*x*x*x*x*x*x
X[:,104] =y*y*y*y*y*y*y*y*y*y*y*y*y
X[:,105] =x*x*x*x*x*x*x*x*x*x*x*x*x*y
X[:,106] =x*x*x*x*x*x*x*x*x*x*x*x*y*y
X[:,107] =x*x*x*x*x*x*x*x*x*x*x*y*y*y
X[:,108] =x*x*x*x*x*x*x*x*x*x*y*y*y*y
X[:,109] =x*x*x*x*x*x*x*x*x*y*y*y*y*y
X[:,110] =x*x*x*x*x*x*x*x*y*y*y*y*y*y
X[:,111] =x*x*x*x*x*x*x*y*y*y*y*y*y*y
X[:,112] =x*x*x*x*x*x*y*y*y*y*y*y*y*y
X[:,113] =x*x*x*x*x*y*y*y*y*y*y*y*y*y
X[:,114] =x*x*x*x*y*y*y*y*y*y*y*y*y*y
X[:,115] =x*x*x*y*y*y*y*y*y*y*y*y*y*y
X[:,116] =x*x*y*y*y*y*y*y*y*y*y*y*y*y
X[:,117] =x*y*y*y*y*y*y*y*y*y*y*y*y*y
X[:,118] =y*y*y*y*y*y*y*y*y*y*y*y*y*y
X[:,119] =x*x*x*x*x*x*x*x*x*x*x*x*x*x

'''
print('X is',X)
X_skl = X[:,1:]
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
k = 5
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
    xTest = xFolds[i]
    #print('x test  is', xTest)
    yTest = yFolds[i]
    #print('y test 1 is', yTest)

    #fit model into (k-1) parts / train set
    #Calculating for ridge
    '''betaRidge = np.linalg.inv(np.transpose(xTrain).dot(xTrain) + (0.015)*np.identity(77)).dot(np.transpose(xTrain)).dot(yTrain)
    ypred = xTest@betaRidge'''
    #linreg = LinearRegression()
    linreg = Lasso(alpha = 1e-6, max_iter = 10e7, tol = 0.0001, fit_intercept = False)
    linreg.fit(xTrain,yTrain)
    ypred=linreg.predict(xTest)
    '''calculate the prediction error of the fitted model when
    predicting the kth part of the data'''

    '''calculate the prediction error of the fitted model when
    predicting the kth part of the data'''

    mse = mean_squared_error(yTest,ypred)
    print('mse = ',mse,' when fold ', (i+1), 'is selected as test data')
    r2Score2 = r2_score(ypred,yTest)

    #print('Variance score:', r2Score )

    print('Variance score:', r2Score2 )
    mseArray.append(mse)
    r2ScoreArray.append(r2Score2)
    #inter = linreg.intercept_
    #interceptArr.append(inter)

print('All the MSEs calculated are:', mseArray)
print('Mean MSE is', np.mean(mseArray))
print('All the r2 scores calculated are:', r2ScoreArray)
print('Mean r2 score is', np.mean(r2ScoreArray))
print('The intercept alpha: \n', interceptArr)
