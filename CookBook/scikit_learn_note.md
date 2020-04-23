# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# ${Introduction\;to\;scikit\;learn(sklearn)}$
---
This notebook demonstartes some of the most useful functions of the beautiful Scitki-Learn library.  
  
What we're going to cover:
  

```python
first = [
    'Clean Data',
    'Transform Data',
    'Reduce Data'   
]

what_were_covering = [  
  "0. An end-to-end Scikit-Learn workflow",
  "1. Getting the data ready",
  "2. Choose the right estimator/algorithm for our problem",
  "3. Fit the model/algorithm and use it to make predictions on our data",
  "4. Evaluating a model",
  "5. Improve a model",
  "6. Save and load a trained model",
  "7. Putting it all together!"]
```

```python
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from IPython.core.display import display, HTML
%matplotlib inline

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()
```

## An end-to-end Scikit_learn workflow

```python
#1. Get the data ready
heart_disease = pd.read_csv('../data/heart-disease.csv')
heart_disease
```

```python
# Create X (features matrix) 
# target 是答案, 所以先drop掉
X = heart_disease.drop('target', axis=1)
X
```

```python
# Create y (labels)
y = heart_disease['target']
y
```

```python
# 2. Choose the right model and hyperparameters
# 基於 a.我們的問題是分類型, 要區分出有心臟並與無心臟並
# 以及 b.我們需要有hyperparameters 來對model進行調整
# 因此使用 RandomForestClassifier (有a、b 兩特性)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# We'll keep the default hyperparameters
clf.get_params()
```

```python
# 3_1. Fit the model to the training data
# 將訓練用的資料與測試用的資料進行分隔
from sklearn.model_selection import train_test_split

# X = heart_disease.drop('target', axis=1) (input)
# y = heart_disease['target'] (output)
# 0.2 = 80% data will be userd for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
```

```python
# X = features/input, y = label/output
clf.fit(X_train, y_train)
```

```python
# 3_2. make a prediction
y_preds = clf.predict(X_test)
y_preds
```

```python
y_test
```

```python
# 4. Evaluate the model on the training data and test data
# 對train data 作評估
clf.score(X_train, y_train)
```

```python
clf.score(X_test, y_test)
```

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_preds))
```

```python
confusion_matrix(y_test, y_preds)
```

```python
accuracy_score(y_test, y_preds)
```

```python
# 5. Improve a model
# Try different amount of n_estimators
np.random.seed(42)
for i in range(10, 100, 10):
    print(f'Trying model with {i} estimators...')
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f'Model accuracy on test set:{clf.score(X_test, y_test) * 100:.2f}%')
```

```python
# RandomForestClassifier.score 與  accuracy_score 的差別再哪
# 1.data ready
# 2.split data > tran, test
# 3.fit(x_tran, y_tran) > clf
# 4.clf.predict(x_test) > y_predict
# 5.clf.score(x_test, y_test)
```

## 1. Getting our data ready to be used with machine learning

Three main things we have to do:
    1. Split the data into features and labels (usully `X` & `y`)
    2. Filling (also called imputing) or disregarding missing values
    3. Converting non-numerical values to numerical values (also called feature encoding)

```python
heart_disease.head()
```

```python
X = heart_disease.drop('target', axis=1)
X
```

```python
y = heart_disease['target']
y.head()
```

```python
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

Clean Data  
移除不完整的資料列, 或填補平均值等等  
Transform Data  
將資料轉換為0、1  
Reduce Data  
過多的資料會造成資源（金錢、時間）的浪費所以縮減數據,也可以稱為降維或列縮減（減去不相關列）  





### 1.1 Make sure it's all numerical

```python
car_sales = pd.read_csv('../data/car-sales-extended.csv')
car_sales.head()
```

```python
car_sales.dtypes
```

```python
# Split into X/y
X = car_sales.drop('Price', axis=1)
y = car_sales['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

```python
# Build machine learning model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

```python

# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 將非數字資料轉換為數字, 其中Doors較為特別, 雖然它已經是數字, 但是幾個車門可以作分類, 因此也加入分類
categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
transformed_x = transformer.fit_transform(X)
pd.DataFrame(transformed_x)
```


<!-- #raw -->
# 還原
dummies = pd.get_dummies(car_sales[['Make', 'Colour', 'Doors']])
dummies
<!-- #endraw -->

```python
# Let's refit the model
# Use transformed_x
X_train, X_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2)
model.fit(X_train, y_train)
```

```python
# 因為汽車的價格無法從顏色、廠牌、車門數來做出準確預測所以分數很低, 
# 但本節重點在於如何將廢數據化的資料轉換為數字
model.score(X_test, y_test)
```

### 1.2 What if where missing values)?
1. Fill them with some value (also known as imputation)
2. Remove the samples with missing data altogether.


```python
# Import car sales misssing data
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing
```

```python
car_sales_missing.isna().sum()
```

```python
# Create X & y
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']

```

```python
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
transformed_x = transformer.fit_transform(X)
pd.DataFrame(transformed_x)
```

### Option 1:Fill missing data with Pandas

```python
# Fill the 'Make' column
car_sales_missing['Make'].fillna('missing', inplace=True)

# Fill the 'Colour' column
car_sales_missing['Colour'].fillna('missing', inplace=True)

# Fill the 'Odometer (KM)' column
car_sales_missing['Odometer (KM)'].fillna(car_sales_missing['Odometer (KM)'].mean(), inplace=True)

# Fill the 'Doors' column
car_sales_missing['Doors'].fillna(4, inplace=True)

```

```python
# Check our dataframe again
car_sales_missing.isna().sum()
```

```python
# Remove row with missing Price value
# 因為 Price 是 label, 當 label缺失時, 將無法準確訓練model
car_sales_missing.dropna(inplace=True)


```

```python
car_sales_missing.isna().sum()
```

```python
len(car_sales_missing)
```

```python
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']
```

```python
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
transformed_x = transformer.fit_transform(car_sales_missing)
transformed_x
```

### Option 2:Fill missing values with Scikit-Learn
