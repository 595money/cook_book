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
將data處理好已進入機器學習  

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
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)
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
# 因為汽車的價格無法從顏色、廠牌、車門數來做出準確預測所以分數很低, 
# 但本節重點在於如何將廢數據化的資料轉換為數字
# model.score(X_test, y_test)
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

# categorical_features = ['Make', 'Colour', 'Doors']
# one_hot = OneHotEncoder()
# transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
# transformed_x = transformer.fit_transform(X)
# pd.DataFrame(transformed_x)
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

```python
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing.head()
```

```python
car_sales_missing.isna().sum()
```

```python
# Drop the rows with no labels
car_sales_missing.dropna(subset=['Price'], inplace=True)

# Split into X ＆ y
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']
```


```python
# Fill missing values with Scilit-Learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numerical values with mean
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
door_imputer = SimpleImputer(strategy='constant', fill_value=4)
num_imputer = SimpleImputer(strategy='mean')

# Define columns
cat_features = ['Make', 'Colour']
door_feature = ['Doors']
num_features = ['Odometer (KM)']

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([('cat_imputer', cat_imputer, cat_features), 
                             ('door_imputer', door_imputer, door_feature), 
                             ('num_imputer', num_imputer, num_features)])

# Transform the data
filled_X = imputer.fit_transform(X)
filled_X
```

```python
car_sales_filled = pd.DataFrame(filled_X, columns=['Make', 'Colour', 'Doors', 'Odometer (KM)'])
car_sales_filled.head()
```

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', 
                                  one_hot, 
                                  categorical_features)],
                               remainder='passthrough')
transformed_X = transformer.fit_transform(car_sales_filled)
transformed_X
```

```python
# Now we've got our data as numbers and filled (no missing values)
# Let's fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```


### Option 2: Filling missing data and transforming categorical data with Scikit-Learn
This notebook updates the code in the "Getting Your Data Ready: Handling Missing Values in Scikit-Learn".

The video shows filling and transforming the entire dataset (X) and although the techniques are correct, it's best to fill and transform training and test sets separately (as shown in the code below).

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing.head()
```

```python
# Check missing values
car_sales_missing.isna().sum()
```

```python
# Drop the rows with no labels
car_sales_missing.dropna(subset=["Price"], inplace=True)
car_sales_missing.isna().sum()
```

Note the difference in the following cell to the videos, the data is split into train and test before any filling missing values or transformations take place.

```python
from sklearn.model_selection import train_test_split

# Split into X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Split data into train and test
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
```

```python
# Check missing values
X.isna().sum()
```

Let's fill the missing values. We'll fill the training and test values separately to ensure training data stays with the training data and test data stays with the test data.

Note: We use fit_transform() on the training data and transform() on the testing data. In essence, we learn the patterns in the training set and transform it via imputation (fit, then transform). Then we take those same patterns and fill the test set (transform only).

```python
# Fill missing values with Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numerical values with mean
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Define columns
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)
])

# Fill train and test values separately
filled_X_train = imputer.fit_transform(X_train)
filled_X_test = imputer.transform(X_test)

# Check filled X_train
filled_X_train
```

Now we've filled our missing values, let's check how many are missing from each set.

```python
# Get our transformed data array's back into DataFrame's
car_sales_filled_train = pd.DataFrame(filled_X_train, 
                                      columns=["Make", "Colour", "Doors", "Odometer (KM)"])

car_sales_filled_test = pd.DataFrame(filled_X_test, 
                                     columns=["Make", "Colour", "Doors", "Odometer (KM)"])

# Check missing data in training set
car_sales_filled_train.isna().sum()
```

```python
# Check missing data in test set
car_sales_filled_test.isna().sum()
```

```python
# Check to see the original... still missing values
car_sales_missing.isna().sum()
```

Okay, no missing values but we've still got to turn our data into numbers. Let's do that using one hot encoding.

Again, keeping our training and test data separate.

```python
# Import OneHotEncoder class from sklearn
from sklearn.preprocessing import OneHotEncoder

# Now let's one hot encode the features with the same code as before 
categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", 
                                 one_hot, 
                                 categorical_features)],
                                 remainder="passthrough")

# Fill train and test values separately
transformed_X_train = transformer.fit_transform(car_sales_filled_train)
transformed_X_test = transformer.transform(car_sales_filled_test)

# Check transformed and filled X_train
transformed_X_train.toarray()
```

### Fit a model
Wonderful! Now we've filled and transformed our data, ensuring the training and test sets have been kept separate. Let's fit a model to the training set and evaluate it on the test set.

```python
# Now we've transformed X, let's see if we can fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor

# Setup model
model = RandomForestRegressor()

# Make sure to use transformed (filled and one-hot encoded X data)
model.fit(transformed_X_train, y_train)
model.score(transformed_X_test, y_test)
```

## 2. Choose the right estimator/algorithm for our problem
根據要處理的問題選擇使用的機器學習模型
* estimator/algorithm = machine learning model in scikit-learn
* Classification - predicting whether a sample is on thing or another.
* Regression - predicting a number
* [scikit-learn ml map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)


### 2.1 Picking a mcachine learning model for a regression problem

```python
# Import Boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
boston
```

```python
boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.Series(boston['target'])
```

```python
boston_df.head()
```

```python
# How many data ?
len(boston_df)
```

```python
# Let's try the Ridge Regression model
from sklearn.linear_model import Ridge

# Setup random seed
np.random.seed(42)

# Create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Ridge model
model = Ridge()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```


How do we improve the score?  
What if Ridge wasn't working?  
Let's refer bsck to the map ... [see also](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

```python
# Let's try the Random Forst Regressor
from sklearn.ensemble import RandomForestRegressor

# Setup random seed
np.random.seed(42)

# Create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Insstatiate Random forset Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Evaluate the Random Forest Regressor
rf.score(X_test, y_test)
```

```python
# Check the ridge model again
model.score(X_test, y_test)
```

### 2.2 Choosing and estimator for a classification problem
Let's go to the map...[see also](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

```python
heart_disease = pd.read_csv('../data/heart-disease.csv')
len(heart_disease)
```

Consulting the map and it says to try `LinearSVC`.

```python
# Import the RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the RandomForestClassifier
clf.score(X_test, y_test)
```


Tidbit:
    1. If you have structured data, used ensemble methods
    2. If you have unstructured data, use deep learning or transfer learning

<!-- #region -->
## 3. Fit the model/algorithm and use it to make predictions on our data  


### 3.1 Fitting the model to the data

Different names for:
* X = features, features variables, data
* y = labels, targets, target variables
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier()

# Fit the model to the data (training the machine learning model)
clf.fit(X_train, y_train)

# Evaluate the RandomForestClassifier (use the patterns the model has learned)
clf.score(X_test, y_test)
```

```python
X.head()
```

### 3.2 Make predictions using a machine learning model

2 way to make predictions:
1. `predict()`
2. `predict_proba()`

```python
# Use a trained model to make predictions
```

```python
clf.predict(X_test)
```

```python
np.array(y_test)
```

```python
# Compare predictions to truth labels to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)
```

```python
clf.score(X_test, y_test)
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)
```

Make predictions with predict_proba()


```python
# predict_proba() returns probabilities of a calssification label
clf.predict_proba(X_test[:5])
```

```python
# Let's predict() on the same data...
clf.predict(X_test[:5])
```

`predict()` can also be used for regression models.

```python
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)

# Create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model
model = RandomForestRegressor().fit(X_train, y_train)

# Make predictions
y_preds = model.predict(X_test)
```


```python
y_preds[:10]
```

```python
# y_test is Answer
np.array(y_test[:10])
```

```python
# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error
from typing import List

#平均絕對誤差 mean absolute error
mean_absolute_error(y_test, y_preds)
```

### 4. Evaluating a machine learning modle

Three ways to evaluate Scikit-Learn models/esitmators:  
評估機器學習模型
1. Estimator `score` method 
2. The `scoring` parameter
3. Problem-specific metric functions.  

### 4.1 Evaluation a model with the score method

```python
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
```

```python
clf.score(X_test, y_test)
```

Let's do the same but for regreesion...

```python
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)

# Create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model
model = RandomForestRegressor().fit(X_train, y_train)
```

```python
model.score(X_test, y_test)
```

### 4.2 Evaluating a model using the `scoring` parameter

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

```python
# clf = train model
# X = feature
# y = label
cross_val_score(clf, X, y)
```

```python
np.random.seed(42)
# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y))

# Compare the two
clf_single_score, clf_cross_val_score
```

```python
# Scoring parameter set to None by default
# 當cross_val_score參數位給scoring時, 預設為None,
# 而 scoring = None (預設), 則會帶入 scoring = clf.score()
# 此處的 clf 是 RandomForestClassifier, 所以預設scoring 是 mean accuracy
# Default scoring parameter of classifier = mean accuracy
# clf.score() = mean accuracy

cross_val_score(clf, X, y, cv=5, scoring=None)
```

### 4.2.1 Classification model evaluation metrics
1. Accuracy
2. Area umder ROC curve
3. Confusion matrix
4. Classification report

**Accuracy**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

clf = RandomForestClassifier()
cross_val_score = cross_val_score(clf, X, y)
```

```python
np.mean(cross_val_score)
```

```python
print(f'Heart Disease Classifier Cross-Validated Accuracy:{np.mean(cross_val_score) * 100:.2f}%')
```

**Area under the receiver operating characteristic curve (AUC/ROC)**
* Area under curve (AUC)
* ROC curve
* [see also](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

ROC curves are a comparison of a model's true postive rate (tpr) versus a models false positive rate (fpr).

* True positive = model predicts 1 when truth is 1 
* False positive = model predicts 1 when truth is 0  
* True negative = model predicts 0 when truth is 0
* Flase negative = model predicts 0 when true is 1

```python
# Create X_test... etc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

```python
from sklearn.metrics import roc_curve

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:10], len(y_probs)
```

```python
y_probs_positive = y_probs[:, 1]
y_probs_positive[:10]
```

```python
# Caculate fpr, tpr and thresholds
# fpr = FP / (FP + TN) 在所有實際陰性樣本中被誤判為陽性的比率, T = 猜對, P = 實際是陽性
# tpr = TP / (TP + FN) 在所有實際陽性樣本中正確判定陽性的比率, F = 猜錯, P = 實際是陽性
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

# Check the false positive rates
fpr
```

```python
# Create a function for plotting ROC curves
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    '''
    Plots a ROC curve given the false positive rate (fpr)
    and true positive rate (tpr) of a model.
    '''
    # Plot roc 
    plt.plot(fpr, tpr, color='orange', label='ROC')
    
    # Plot line with no predictive power (baseline) 繪製一條沒有進行預測的基準線來比較
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
    
    # Customize the plot
    plt.xlabel('False positive rate (fpr)')
    plt.ylabel('True positive rate (tpr)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr, tpr)



```

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_probs_positive)
```

```python
# Plot perfect ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve(fpr, tpr)
```

```python
# Perfect AUC score
roc_auc_score(y_test, y_test)
```

**Confusion Matrix**  
A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.  
混淆矩陣是比較模型預測的標籤和模型應預測的實際標籤的快速方法。
  
In essence, giving you an idea of where the model is getting confused.  
從本質上講，讓您了解模型的混亂之處

```python
from sklearn.metrics import confusion_matrix

# y_test = test sets 的 label
# y_preds = 使用 X_test 進行預測得到的 y_preds
y_preds = clf.predict(X_test)
confusion_matrix(y_test, y_preds)
```

```python
# Visualize confusion matrix with pd.crosstab()
pd.crosstab(y_test,
           y_preds,
           rownames=['Actual Labels'],
           colnames=['Predicted Labels'])
```

```python
# install seaborn
# How install a conda package into the current envrionment from a Jupyter Notebook
# import sys
# !conda install --yes --prefix {sys.prefix} seaborn
```

```python

```

```python
# Make our confusion matrix more visual with Seaborn's heatmp()
import seaborn as sns

# Set the font scale
sns.set(font_scale=1.5)

# Create a confusion martix
conf_mat = confusion_matrix(y_test, y_preds)

# Plot it using Seaborn
sns.heatmap(conf_mat);

```

```python
def plot_conf_mat(conf_mat):
    '''
    Plots a confusion matrix ustin Seaborn's heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(conf_mat,
                    annot=True, # Annotate the boxes with conf_mat info
                     cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label');

plot_conf_mat(conf_mat)
    
```

```python
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test);
```

**Classification report**

```python
from sklearn.metrics import classification_report

# y_test = test sets 的 label
# y_preds = 使用 X_test 進行預測得到的 y_preds
# 將y_test 與 y_preds進行比較
print(classification_report(y_test, y_preds))

```

1:41 記得重看
Precision  
Recall  
F1-score  
Suppot  
Accuracy  
Macro avg  
Weighter abg  
