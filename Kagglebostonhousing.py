'''
Project - Perform Multiple Linear Regression and predict MEDV
from Boston Housing Dataset BostonData.csv

The Boston Housing Dataset is a derived from information
collected by the U.S. Census Service concerning housing
in the area of Boston MA.

The following describes the dataset columns:

CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots 
over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds
river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built
prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of
blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's

This dataset is also included in many python distribution
like anaconda however I am using the dataset downloaded
from kaggle. So let's code

# Prerequisites
# - Knowledge of basic Python
# - Statistics
# - Data Processing
# - Multiple Linear regression

'''
# ----------------------------------------------
# Step 0 - Import Libraries
# ----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# ----------------------------------------------
# Step 1 - Read the data
# ----------------------------------------------
dataset = pd.read_csv('BostonData.csv')
dataset.head()

# ----------------------------------------------
# Step 2 - Prelim Analysis and Feature selection
# ----------------------------------------------
#create a new dataset and remove the sl
# so that the original dataset is as it is.
data_prep = dataset.copy()
data_prep = data_prep.drop(['sl'], axis = 1)


# ----------------------------------------------
# Step 3 - Data Visualisation (Exploratory Data Analysis)
# ----------------------------------------------
# let's check for missing values in the dataset
data_prep.isnull().sum()

# Create pandas histogram
data_prep.hist(rwidth = 0.9)
plt.tight_layout()

sns.set(rc={'figure.figsize':(10,5)})
sns.distplot(data_prep['medv'], bins=30)
plt.show()

#create a correlation matrix
correlation_matrix = data_prep.corr().round(2)
#annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

#plot graphs against medv with all other features
from sklearn import preprocessing
# scale the features using minmax scalar before plotting them against medv
min_max_scaler = preprocessing.MinMaxScaler()
columns = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age', 'crim']
x = data_prep.loc[:,columns]
y = data_prep['medv']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=columns)
fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(columns):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# let's Split the data into training and testing sets in 80:20 ratio
from sklearn.model_selection import train_test_split

X = data_prep.drop(['rad','medv'], axis=1)
Y = data_prep['medv']

# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#train the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linReg_model = LinearRegression()
linReg_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = linReg_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = linReg_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
























