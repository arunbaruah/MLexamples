# Compare the effect of Lasso, Ridger and Linear Regression
# for presence of Multicollinearity as well as their effect on
# coefficients of different features
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet

df = pd.read_csv('multicollinearityL1L2data.csv')
x = df.iloc[:, :-1]
y = df.iloc[:,  -1]

# Check and confirm the presence of Multicollinearity
correlation = df.corr()

# Perform Linear Regression 
linearReg = LinearRegression()
linearReg.fit(x, y)
linearReg_coeff = linearReg.coef_
linearReg_intercept = linearReg.intercept_

# Perform Lasso regression, take random alpha in this case 10
lassoReg = Lasso(alpha=10)
lassoReg.fit(x, y)
lasso_coeff = lassoReg.coef_
lasso_intercept = lassoReg.intercept_

# Perform Ridge Regression. Take a very high alpha value so that
# we can easily vasualize the changes.
ridgeReg = Ridge(alpha=100)
ridgeReg.fit(x, y)
ridge_coeff = ridgeReg.coef_
ridge_intercept = ridgeReg.intercept_



# compare the values of three lists of coefficients





