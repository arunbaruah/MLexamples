import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

df = pd.read_csv('ridgeRegData.csv')

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

linearReg = LinearRegression()
linearReg.fit(x,y)
linearReg_coefficient = linearReg.coef_
linearReg_intercept = linearReg.intercept_

plt.scatter(x, y, color = "red")
plt.plot(x, linearReg.predict(x), color = "green")
plt.title("X vs Y (Training set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

alphaval = 0
ridge = Ridge(alpha=alphaval)
ridge.fit(x,y)
ridge_coefficient = ridge.coef_
ridge_intercept = ridge.intercept_

plt.scatter(x, y, color = "red")
plt.plot(x, ridge.predict(x), color = "green")
plt.title("X vs Y (Applying Ridge Regression - Training set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Now let's change the aplha value with a different value and see the effect
# Now I will copy some of the above code so that we can make changes here 
# and change the alpha values and we can select the below code and run
# First set the alpha to 1 then run the code, repeat for 2, 3, 4


# Now close your graphic window. In case you cannot see the graph window 
#separately then  
'''
Start by opening the Spyder preferences.
Mac users can go to python -> Preferences... in the menubar
Linux/Windows users can go to Tools -> Preferences
In the Preferences window, click on IPython console on the left side of the 
window, then on the Graphics tab.
Under Graphics backend, select Automatic for the backend.
Restart Spyder.

'''
#alphaval = 0, 1, 2, 3, 4 one by one
alphaval = 0
ridge = Ridge(alpha=alphaval)
ridge.fit(x,y)
ridge_coefficient = ridge.coef_
ridge_intercept = ridge.intercept_

x_plt = [0,1,2,3,4]
y_plt = ridge.predict(pd.DataFrame(x_plt))
#we can change this value

plt.figure(2)
plt.plot(x_plt, y_plt)

plt.ylim(ymin=0, ymax =9)
plt.xlim(xmin=0, xmax =6)

# y = mx + b
# for \u03BB or \u03B1 = ' are for printing lambda or alpha special char
plt.text(x_plt[-1], y_plt[-1],
         ' y = '                          +
         str('%.2f' %ridge_coefficient)   +
         ' * x + '                        +
         str('%.2f' %ridge_intercept)      +
         '   for \u03BB or \u03B1 = '     +
         str(alphaval),
         fontsize=8
        )

#Changing the alphaval and run the again and again is headace 
#so could use a for loop so lets rewrite the above code
# Create test data for plotting
# Create a list of different alpha/penalty parameter values
alphaval2 = [0,1,5,10,20]

# Plot different Ridge Regression lines in one figure
for i, alphaval in enumerate(alphaval2):
    ridge = Ridge(alpha=alphaval)
    ridge.fit(x, y)
    ridge_coefficient = ridge.coef_
    ridge_intercept = ridge.intercept_
    y_plt = ridge.predict(pd.DataFrame(x_plt))
    
    plt.figure(1)       
    plt.plot(x_plt, y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)
    plt.text(x_plt[-1], y_plt[-1],  
             ' y = ' + 
             str('%.2f' %ridge_coefficient) +
             ' * x' + 
             ' + ' + 
             str('%.2f' %ridge_intercept) +
             '      for \u03BB or \u03B1 = ' + str(alphaval), fontsize=10)


























