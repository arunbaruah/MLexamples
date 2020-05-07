#Build the Logistic Regression Model

#import the data
import pandas as pd
#read the data
dataset = pd.read_csv("LoanDataset-1.csv")
#check for null value
dataset.isnull().sum(axis=0)

# Replace Missing Values. Drop the rows.
dataset = dataset.dropna()

# Drop the column gender as we do not need it in this example
dataset = dataset.drop(['gender'], axis=1)

# Create Dummy variables
dataset.dtypes
dataset = pd.get_dummies(dataset, drop_first=True)

# Normalize Income and Loan Amount using StandardScaler
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()

dataset['income'] = scalar_.fit_transform(dataset[['income']])
dataset['loanamt'] = scalar_.fit_transform(dataset[['loanamt']])

# Create the X  and Y 
Y = dataset[['status_Y']]
X = dataset.drop(['status_Y'], axis=1)

# Split the X and Y dataset into trai test set in 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

# Build the model  
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, Y_train)

# Predict the outcome using Test data
Y_predict = logistic_reg.predict(X_test)

# Build the conufsion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)

score = logistic_reg.score(X_test, Y_test)

