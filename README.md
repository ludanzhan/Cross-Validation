# Cross-Validation
Using Cross Validation to fit the best model. Visualize data with scatter plot and seasonal chart.
## All Libraries imported:
```python
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures
```

## Steps in finding the best model
1. The goal of this project is to fit the best model with the smallest mean absolute percentage error, first create a function that returns mean absolute percentage error
```python
def mean_absolute_percentage_error(y,y_pred): y = np.array(y)
y_pred = np.array(y_pred)
return 100*np.mean(np.abs((y-y_pred)/y))
```
2. Fit the model with a single predictior and use scattor plot to determine whether an additional predictor need to be added
```python
temp = df.temp
temp1 = temp.values.reshape(-1,1)
model1 = LinearRegression().fit(temp1,load)
yhat = model1.predict(temp1)

plt.figure(figsize = (40,20))
plt.scatter(temp,load,c='b',s=10)
plt.plot(temp,yhat,c='r',lw=3)
plt.xlabel('temp')
plt.ylabel('load')
```



