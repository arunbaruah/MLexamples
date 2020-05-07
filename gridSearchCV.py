# Compare multiple Classifiers and tune the hyperparameters using GridSearchCV
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Read dataset
data = pd.read_csv('adultincomedataset.csv')
data_prep = pd.get_dummies(data, drop_first=True)
x = data_prep.iloc[:, :-1]
y = data_prep.iloc[:, -1]
rfc = RandomForestClassifier(random_state=123)
svc = SVC(random_state=123)

# Define parameters for Random Forest
randomForest_param = {'n_estimators':[10,15,20], 
            'min_samples_split':[8,16],
            'min_samples_leaf':[1,2,3,4,5]
            }

# The parameters results in 3 x 2 x 5 = 30 different combinations
# CV=10 for 30 different combinations mean 300 jobs/model runs
# n_jobs=-1 we are telling the program to use all available CPU cores & threads
randomForest_grid = GridSearchCV(estimator=rfc, 
                        param_grid=randomForest_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search
randomForest_grid_fit = randomForest_grid.fit(x, y)

# Get the results of the GridSearchCV
cv_results_randomForest = pd.DataFrame.from_dict(randomForest_grid_fit.cv_results_)

# define parameters for Support Vector Classifier
supportVector_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10], 
            'kernel':['rbf', 'linear'],
            'gamma':[0.1, 0.25, 0.5, 1, 5]
            }

# The parameters results in 7 x 2 x 5 = 70 different combinations
# CV=10 for 70 different combinations mean 700 jobs/model runs

supportVector_grid = GridSearchCV(estimator=svc, 
                        param_grid=supportVector_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search for Support Vector
supportVector_grid_fit = supportVector_grid.fit(x, y)

# Get the Grid Search results for Support Vector
cv_results_svc = pd.DataFrame.from_dict(supportVector_grid_fit.cv_results_)

# Get the top ranked test score for all the three classifiers
randomForest_top_rank = cv_results_randomForest[cv_results_randomForest['rank_test_score'] == 1]
supportVector_top_rank = cv_results_svc[cv_results_svc['rank_test_score'] == 1]


# Print the train and test score for three classifiers

print('\n\n')

print ('                    ',
       '  Random Forest    ',
       '  Support Vector   ')
print ('                    ',
       '  ---------------- ',
       '  ---------------- ')
print ('  Mean Test Score   : ', 
       str(randomForest_top_rank['mean_test_score']),
       '            ',
       str(supportVector_top_rank['mean_test_score']),
       '                '
       )

print ('  Mean Train Score  : ', 
       str(randomForest_top_rank['mean_train_score']),
       '            ',
       str(supportVector_top_rank['mean_train_score']),
       '                '
       )




