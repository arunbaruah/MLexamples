# Cross validation example
# import all the libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
# Read dataset
ds = pd.read_csv('adultincomedataset.csv')
# Create Dummy variables
data_prep = pd.get_dummies(ds, drop_first=True)
# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]
# Train Classifiers
dtc = DecisionTreeClassifier(random_state=123)
rfc = RandomForestClassifier(random_state=123)
svc = SVC(kernel='rbf', gamma=0.5)
# Perform cross validation and store the results
cv_results_dtc = cross_validate(dtc, X, Y, cv=10, return_train_score=True)
cv_results_rfc = cross_validate(rfc, X, Y, cv=10, return_train_score=True)
cv_results_svc = cross_validate(svc, X, Y, cv=10, return_train_score=True)
# Average of all the results
dtc_test_average = np.average(cv_results_dtc['test_score'])
rfc_test_average = np.average(cv_results_rfc['test_score'])
svc_test_average = np.average(cv_results_svc['test_score'])

dtc_train_average = np.average(cv_results_dtc['train_score'])
rfc_train_average = np.average(cv_results_rfc['train_score'])
svc_train_average = np.average(cv_results_svc['train_score'])

# print the results 
print('\n')
print('\n')
print('\t','Decision Tree  ', 'Random Forest  ','Support Vector   ')
print('\t','---------------', '---------------','-----------------')

print('Test  : ',
      round(dtc_test_average, 3), '\t\t',
      round(rfc_test_average, 3), '\t\t',
      round(svc_test_average, 3))

print('Train : ',
      round(dtc_train_average, 3), '\t\t',
      round(rfc_train_average, 3), '\t\t',
      round(svc_train_average, 3))




