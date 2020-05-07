# Implement various feature selection, Select Transforms
import pandas as pd
f = pd.read_csv('Students-dataset-03.csv')
X = f.iloc[:, :-1]
Y = f.iloc[:,  -1]
# Import various select transforms along with the f_regression mode
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
# Implement GenericUnivariateSelect with k_best
selectorG1 = GenericUnivariateSelect(score_func=f_regression,
                                     mode='k_best',
                                     param=3)
x_g1 = selectorG1.fit_transform(X,Y)
# Get f_score and p_values for the selected features
f_score = selectorG1.scores_
p_values = selectorG1.pvalues_
# Print the f_score and p_values
# Print the table of Features, F-Score and P-values
columns = list(X.columns)
print ("\n\n ")
print ("    Features     ", "F-Score    ", "P-Values")
print ("    -----------  ---------    ---------")
for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("    ", columns[i].ljust(12), f1.rjust(8),"    ", p1.rjust(8))

cols = selectorG1.get_support(indices=True)
selectedCols = X.columns[cols].to_list()
print(selectedCols)












