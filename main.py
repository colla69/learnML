# Load CSV using Pandas from URL
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)

# print a Description of the CSV dataset
description = data.describe()
# print(description)

# create and disply a Scatter Plot Matrix# scatter_matrix(data)
# plt.show()


# Standardize a Dataset.
# the snippet below loads the dataset, calculates
# the parameters needed to standardize the data, then creates a standardize copy of the input
#  data.
dataframe = read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
# print(rescaledX[0:5, :])

# Alg Evaluation with resampling Mehtods
# we need to split a dataset in treining and test sets
# estimate the accuracy using k-fold cross-validation
kfold = KFold(n_splits=10, random_state=7)  # ??
model = LogisticRegression()  # ??
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
# practice Estimating the accuracy of an algorithm using leave one out cross-validation.

# alg evaluation metrics
# Practice using the Accuracy and LogLoss metrics on a classification problem.
# Practice generating a confusion matrix and a classification report
# Practice using RMSE and RSquared metrics on a regression problem.
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))

#  Spot-check linear algorithms on a dataset (e.g. linear regression, logistic regression and
# linear discriminate analysis).
# Spot-check some nonlinear algorithms on a dataset (e.g. KNN, SVM and CART).
#  Spot-check some sophisticated ensemble algorithms on a dataset (e.g. random forest and
# stochastic gradient boosting).
# KNN Regression
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("nonlinear regression check " + str(results.mean()))  # Spot-Check a Nonlinear Regression Algorithm.

# how to check more models at once
# prepare models
models = []
models.append(('LR', LogisticRegression()))  # ??
models.append(('LDA', LinearDiscriminantAnalysis()))  # ??
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# improving performance
# Tune the parameters of an algorithm using a grid search that you specify
# Tune the parameters of an algorithm using a random search.
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()  # ??
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

