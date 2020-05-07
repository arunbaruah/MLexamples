# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the data
dataset = pd.read_csv('BikeShareData.csv')

# Now let's do the Prelim Analysis and Feature selection
#care in this example I am directly modifying the existing dataset
#you can also create a separate variable and copy the required data
#axis =1 means the columns
dataset = dataset.drop(['date', 'casual', 'registered'], axis=1)

#let checks if any missing values exists in the dataset
dataset.isnull().sum()

# Let's create histogram. rwidth = 0.9 is the width of the bar
dataset.hist(rwidth = 0.9)
plt.tight_layout()

# Visualise the continuous features Vs demand
# I am not using any loops here keeping the beginners in mind
# so subplot one by one. We can also do this with just one function
plt.subplot(2,2,1)
plt.title('Temperature Vs Demand')
plt.scatter(dataset['temp'], dataset['demand'], s=2, c='g')

plt.subplot(2,2,2)
plt.title('aTemp Vs Demand')
plt.scatter(dataset['atemp'], dataset['demand'], s=2, c='b')

plt.subplot(2,2,4)
plt.title('Windspeed Vs Demand')
plt.scatter(dataset['windspeed'], dataset['demand'], s=2, c='c')

plt.subplot(2,2,3)
plt.title('Humidity Vs Demand')
plt.scatter(dataset['humidity'], dataset['demand'], s=2, c='m')

# Visualise the categorical features
colors = ['g', 'r', 'm', 'b']
# we have season, month, holiday, weekday, year, hour, workingday 
# and weather so we create a subplot of 3 by 3 
plt.subplot(3,3,1)
plt.title('Average Demand per Season')
uniqueSeason = dataset['season'].unique()
averageSeason = dataset.groupby('season').mean()['demand']
plt.bar(uniqueSeason, averageSeason, color=colors)

plt.subplot(3,3,2)
plt.title('Average Demand per month')
uniqueMonth = dataset['month'].unique()
averageMonth = dataset.groupby('month').mean()['demand']
plt.bar(uniqueMonth, averageMonth, color=colors)

plt.subplot(3,3,3)
plt.title('Average Demand per Holiday')
uniqueHoliday = dataset['holiday'].unique()
averageHoliday = dataset.groupby('holiday').mean()['demand']
plt.bar(uniqueHoliday, averageHoliday, color=colors)

plt.subplot(3,3,4)
plt.title('Average Demand per Weekday')
uniqueWeekday = dataset['weekday'].unique()
averageWeekday = dataset.groupby('weekday').mean()['demand']
plt.bar(uniqueWeekday, averageWeekday, color=colors)

plt.subplot(3,3,5)
plt.title('Average Demand per Year')
uniqueYear = dataset['year'].unique()
averageYear = dataset.groupby('year').mean()['demand']
plt.bar(uniqueYear, averageYear, color=colors)

plt.subplot(3,3,6)
plt.title('Average Demand per hour')
uniqueHour = dataset['hour'].unique()
averageHour = dataset.groupby('hour').mean()['demand']
plt.bar(uniqueHour, averageHour, color=colors)

plt.subplot(3,3,7)
plt.title('Average Demand per Workingday')
uniqueWorkingday = dataset['workingday'].unique()
averageWorkingday = dataset.groupby('workingday').mean()['demand']
plt.bar(uniqueWorkingday, averageWorkingday, color=colors)

plt.subplot(3,3,8)
plt.title('Average Demand per Weather')
uniqueWeather = dataset['weather'].unique()
averageWeather = dataset.groupby('weather').mean()['demand']
plt.bar(uniqueWeather, averageWeather, color=colors)

# Check for outliers as well as what percentile they exist
dataset['demand'].describe()

#intervals -- 0.05, 0.1, 0.15, 0.9, 0.95, 0.99
dataset['demand'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])

# Linearity using correlation coefficient matrix using corr
correlation_matrix = dataset[['temp', 'atemp', 'humidity', 'windspeed', 'demand']].corr()

# Drop irrelevant features
dataset = dataset.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)

# Autocorrelation of demand using acor
# we need to change the datatype to float so create a separate variable
# maxlags=12 because we are looking at 24 hrs time
temp = pd.to_numeric(dataset['demand'], downcast='float')
plt.acorr(temp, maxlags=12)

# Log Normalise the feature 'Demand' and create 2 variable
temp1 = dataset['demand']
temp2 = np.log(temp1)

#now plot both of these one by one
plt.figure()
temp1.hist(rwidth=0.9, bins=20)

plt.figure()
temp.hist(rwidth=0.9, bins=20)

#now let's modify the actual column
dataset['demand'] = np.log(dataset['demand'])

# Solve the problem of Autocorrelation
# Shift the demand by 3 lags

lag_1 = dataset['demand'].shift(+1).to_frame()
lag_1.columns = ['lag_1']

lag_2 = dataset['demand'].shift(+2).to_frame()
lag_2.columns = ['lag_2']

lag_3 = dataset['demand'].shift(+3).to_frame()
lag_3.columns = ['lag_3']

#create a new dataset having the lags and the original dataset
dataset_with_lag = pd.concat([dataset, lag_1, lag_2, lag_3], axis=1)

#finally drop the na values
dataset_with_lag = dataset_with_lag.dropna()

# Create Dummy Variables and drop first to avoid dummy variables trap
# let's check the datatypes
dataset_with_lag.dtypes

# change the datatypes
dataset_with_lag['season'] = dataset_with_lag['season'].astype('category')
dataset_with_lag['holiday'] = dataset_with_lag['holiday'].astype('category')
dataset_with_lag['weather'] = dataset_with_lag['weather'].astype('category')
dataset_with_lag['month'] = dataset_with_lag['month'].astype('category')
dataset_with_lag['hour'] = dataset_with_lag['hour'].astype('category')

dataset_with_lag.dtypes

#create dummies now
dataset_with_lag = pd.get_dummies(dataset_with_lag, drop_first=True)

# Create Train and test split in 80:20 ratio
# take care the demand has a time dependency so follow the below pattern

Y = dataset_with_lag[['demand']]
X = dataset_with_lag.drop(['demand'], axis=1)

tr_size = 0.7 * len(X)
tr_size = int(tr_size)

X_train = X.values[0 : tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0 : tr_size]
Y_test = Y.values[tr_size : len(Y)]

#Fit and Score the model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

r_squared_train = lin_reg.score(X_train, Y_train)
r_squared_test  = lin_reg.score(X_test, Y_test)

# Create Y Predictions
Y_predict = lin_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))

# Calculate RMSLE and compare results
# Note that we had previously transformed the demand into log values
# now we need to convert back to exponent it's opposed to log
# before we calculate the RMSLE
# so we need to convert the Y_test and Y_predict

Y_test_exp = []
Y_predict_exp = []

for i in range(0, len(Y_test)):
    Y_test_exp.append(math.exp(Y_test[i]))
    Y_predict_exp.append(math.exp(Y_predict[i]))

#now calculate the rmsle as per the formula
log_sq_sum = 0.0
for i in range(0, len(Y_test_exp)):
    log_a = math.log(Y_test_exp[i] + 1)
    log_p = math.log(Y_predict_exp[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff

rmsle = math.sqrt(log_sq_sum/len(Y_test))

print(rmsle)
































