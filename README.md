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
![scatter plot](https://github.com/ludanzhan/Cross-Validation/blob/main/scatter%20plot.png)
The image above suggesting an additional predictor to be added.

3. Using cross validation to compare all potential models
  - Spliting the data into train set and test set
  ``` python
   df_2019 = df[df['year']==2019]
   y_test = df_2019.Load
   x_test = df_2019.drop(columns = ['Load','Date','prediction','squareTemp'], axis= 1)
   
   df_train = df[df['year']!=2019]
   df_train = df_train.dropna()
   df_train = df_train.drop(columns = ['squareTemp','prediction'], axis = 1)
   df_train[:5]
  ```
  - Using _itertools_ to find all possible combination of predictors
  ```python
   num = len(x_train.columns)
   for k in range(1,num+1):
   for subset in itertools.combinations(x_train.columns,k):
  ```
  - fit the model to the test set and repot the mean absolute percentage error
  ```python
   features.append(subset)
   num_feature.append(len(subset))
   
   x_train2 = x_train[list(subset)]
   x_test2 = x_test[list(subset)]
   
   model = LinearRegression().fit(x_train2,y_train)
   yhat = model.predict(x_test2)
   R_squared = model.score(x_train2,y_train)
   mape = mean_absolute_percentage_error(y_test,yhat)
   
   MAPE.append(mape)
   r2.append(R_squared)
  ```
  - Find the smllest mean absolute percentage error in each numbers of combination and choose the smallest one as the best model
  ``` python
   best_mape = data.groupby(['num_features'])['MAPE'].min()
   data2 = pd.DataFrame() for i in range(1,8):
   ata2 = data2.append(data[data['MAPE'] == best_mape[i]])
  ```
4. Using _"Seasonal Chart"_ to summary each year's data
![image](https://github.com/ludanzhan/Cross-Validation/blob/main/seasonal%20chart.png)


