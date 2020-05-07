import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
df = pd.read_csv('ridgeRegData.csv')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create a list of different alpha/penalty parameter values
lassoVal = [0,0.5,1,2,4]
lasso = Lasso(alpha=0)
lasso.fit(x,y)
lasso_coefficient = lasso.coef_
lasso_intercept = lasso.intercept_
x_plt = [0,1,2,3,4]
y_plt = lasso.predict(pd.DataFrame(x_plt))
plt.figure(1)
plt.plot(x_plt, y_plt)
plt.ylim(ymin=0, ymax =9)
plt.xlim(xmin=0, xmax =6)

# Plot different lasso Regression lines in one figure
for i, lasso_val in enumerate(lassoVal):
    lasso = Lasso(alpha=lasso_val)
    lasso.fit(x, y)
    lasso_coefficient = lasso.coef_
    lasso_intercept = lasso.intercept_
    y_plt = lasso.predict(pd.DataFrame(x_plt))
    
    plt.figure(1)       
    plt.plot(x_plt, y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)
    plt.text(x_plt[-1], y_plt[-1],  
             ' y = ' + 
             str('%.2f' %lasso_coefficient) +
             ' * x' + 
             ' + ' + 
             str('%.2f' %lasso_intercept) +
             '      for \u03BB or \u03B1 = ' + str(lasso_val), fontsize=10)


























