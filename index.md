## Lab 4

Here is the code I used for the coding section of Lab 4. I'm not yet sure why I got Q9 wrong but I will update this when I figure it out.

### Imports and Loading the data
The code below loads and imports the Boston Housing data, along with creating the matrix X_scaled to hold the scaled features:

```python
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as SS
from sklearn.pipeline import Pipeline

mpl.rcParams['figure.dpi'] = 350
%matplotlib inline
%config InlineBackend.figure_format = 'retina'


data = load_boston()
Xdf = pd.DataFrame(data=data.data, columns = data.feature_names)
y = data.target

ss = SS()
X_scaled = ss.fit_transform(Xdf) # Z-scores
```

## Question 6

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model = model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE for linear regression with all features: ", rmse)
```
This produced the Output:

RMSE for linear regression with all features:  4.679191295697281

## Question 7

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

n_folds = 10
kf = KFold(n_splits=n_folds, random_state=1234,shuffle=True)

PE = []
model = Lasso(alpha=0.03)

for train_index, test_index in kf.split(X_scaled):
    X_train = X_scaled[train_index]
    y_train = y[train_index]
    X_test = X_scaled[test_index]
    y_test = y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    
print("\n10-fold cross-validation Lasso prediction error: ", np.mean(PE))
```
This produced the Output:

10-fold cross-validation Lasso prediction error:  4.786790456607522

## Question 8

```python
n_folds = 10
kf = KFold(n_splits=n_folds, random_state=1234,shuffle=True)

PE = []
model = ElasticNet(alpha=0.05, l1_ratio=0.9)

for train_index, test_index in kf.split(X_scaled):
    X_train = X_scaled[train_index]
    y_train = y[train_index]
    X_test = X_scaled[test_index]
    y_test = y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    
print("\n10-fold cross-validation Elastic Net prediction error: ", np.mean(PE))
```
This produced the Output:

10-fold cross-validation Elastic Net prediction error:  4.785480197221984


## Question 9

```python
from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=2)
X_scaled_poly = polynomial_features.fit_transform(X_scaled)

model = LinearRegression()
model.fit(X_scaled_poly, y)
y_pred_poly = model.predict(X_scaled_poly)

rmse = np.sqrt(mean_squared_error(y, y_pred_poly))
print("RMSE for polynomial regression with all features: ", rmse)
```
This produced the Output:

RMSE for polynomial regression with all features:  2.449087064744557

## Question 10

```python
polynomial_features = PolynomialFeatures(degree=2)
X_scaled_poly = polynomial_features.fit_transform(X_scaled)

model = Ridge(alpha=0.1)
model.fit(X_scaled_poly, y)

residuals = (y - model.predict(X_scaled_poly))
```
```python
import pylab
import statsmodels.api as sm
sm.qqplot(residuals, loc = 0, scale = 1, line='s')
pylab.show()
```
