# Decision Tree Classifier
# Predict the income of an adult based on the census data
# Import libraries
import pandas as pd

# Read dataset
dataset = pd.read_csv('adultincomedataset.csv')

# Check for Null values
dataset.isnull().sum(axis=0)

# Create Dummy variables
dataset.dtypes
data_prep = pd.get_dummies(dataset, drop_first=True)


# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]


# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 123, stratify=Y)


# Import and train classifier
from sklearn.tree import DecisionTreeClassifier
decesionTreeClassifierModel = DecisionTreeClassifier(random_state=123)
decesionTreeClassifierModel.fit(X_train, Y_train)


# Test the model
Y_predict = decesionTreeClassifierModel.predict(X_test)

# Evaluate the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = decesionTreeClassifierModel.score(X_test, Y_test)






