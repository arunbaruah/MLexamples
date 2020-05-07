#standardize the features using StandardScaler needs few steps
#let's download iris dataset and perform the standardization
import pandas as pd
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.head()

# split data table into data X and class labels y
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

#standardize the features using StandardScaler
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

#continuing with the same program and the iris dataset
import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#instead of line#24 and 25, we could directly use the numpy function np.cov
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

#Compute the eigenvectors of this covariance matrix
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Selecting the principal component analysis
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#calculate the explained variance from the eigenvalues
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.plot(np.cumsum(var_exp))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');







































