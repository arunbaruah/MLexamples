#Build the Logistic Regression Model Evaluation Matrix
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
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_predict)
score = logistic_reg.score(X_test, Y_test)
classificationReport = classification_report(Y_test, Y_predict)
#You can also use this function to get the accuracy score
from sklearn.metrics import accuracy_score
accuracyScore = accuracy_score(Y_test, Y_predict)

#Lines for Hands on - Adjusting Thresholds
#Create prediction probability
y_prob = logistic_reg.predict_proba(X_test)

#let's modify line 44
y_prob1 = logistic_reg.predict_proba(X_test)[:, 1]

y_new_pred = []
#increate the threshold to .8
threshold = 0.75

for i in range(0,len(y_prob1)):
    if y_prob1[i] > threshold:
        y_new_pred.append(1)
    else:
        y_new_pred.append(0)

#confusion matrix,scores and classification report
confusionMatrix2 = confusion_matrix(Y_test, y_new_pred)
score2 = accuracy_score(Y_test, y_new_pred)
classificationReport1 = classification_report(Y_test, y_new_pred) 

#new threshold value now take 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 like that
# and re-run the above lines from line# 49 to 62

#understand and implement AUC ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

#declare variable fpr (false positive rate) tpr (true +ve rates) and threshold
fpr, tpr, threshold = roc_curve(Y_test, y_new_pred)

auc = roc_auc_score(Y_test, y_new_pred)

#plot the ROC
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, linewidth=5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Loan Application Approval Prediction")
plt.grid()












