# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ${Introduction\;to\;scikit\;learn(sklearn)}$
# ---
# This notebook demonstartes some of the most useful functions of the beautiful Scitki-Learn library.  
#   
# What we're going to cover:
#   

# %%
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

# %%
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from IPython.core.display import display, HTML
# %matplotlib inline

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()

# %% [markdown]
# ## An end-to-end Scikit_learn workflow

# %%
#1. Get the data ready
heart_disease = pd.read_csv('../data/heart-disease.csv')
heart_disease

# %%
# Create X (features matrix) 
# target 是答案, 所以先drop掉
X = heart_disease.drop('target', axis=1)
X

# %%
# Create y (labels)
y = heart_disease['target']
y

# %%
# 2. Choose the right model and hyperparameters
# 基於 a.我們的問題是分類型, 要區分出有心臟並與無心臟並
# 以及 b.我們需要有hyperparameters 來對model進行調整
# 因此使用 RandomForestClassifier (有a、b 兩特性)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# We'll keep the default hyperparameters
clf.get_params()

# %%
# 3_1. Fit the model to the training data
# 將訓練用的資料與測試用的資料進行分隔
from sklearn.model_selection import train_test_split

# X = heart_disease.drop('target', axis=1) (input)
# y = heart_disease['target'] (output)
# 0.2 = 80% data will be userd for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# %%
# X = features/input, y = label/output
clf.fit(X_train, y_train)

# %%
# 3_2. make a prediction
y_preds = clf.predict(X_test)
y_preds

# %%
y_test

# %%
# 4. Evaluate the model on the training data and test data
# 對train data 作評估
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_preds))

# %%
confusion_matrix(y_test, y_preds)

# %%
accuracy_score(y_test, y_preds)

# %%
# 5. Improve a model
# Try different amount of n_estimators
np.random.seed(42)
for i in range(10, 100, 10):
    print(f'Trying model with {i} estimators...')
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f'Model accuracy on test set:{clf.score(X_test, y_test) * 100:.2f}%')

# %%
# RandomForestClassifier.score 與  accuracy_score 的差別再哪
# 1.data ready
# 2.split data > tran, test
# 3.fit(x_tran, y_tran) > clf
# 4.clf.predict(x_test) > y_predict
# 5.clf.score(x_test, y_test)

# %% [markdown]
# ## 1. Getting our data ready to be used with machine learning
#
# Three main things we have to do:
#     1. Split the data into features and labels (usully `X` & `y`)
#     2. Filling (also called imputing) or disregarding missing values
#     3. Converting non-numerical values to numerical values (also called feature encoding)

# %%
heart_disease.head()

# %%
X = heart_disease.drop('target', axis=1)
X

# %%
y = heart_disease['target']
y.head()

# %%
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# Clean Data  
# 移除不完整的資料列, 或填補平均值等等  
# Transform Data  
# 將資料轉換為0、1  
# Reduce Data  
# 過多的資料會造成資源（金錢、時間）的浪費所以縮減數據,也可以稱為降維或列縮減（減去不相關列）  
#
#
#

# %% [markdown]
# ### 1.1 Make sure it's all numerical

# %%
car_sales = pd.read_csv('../data/car-sales-extended.csv')
car_sales.head()

# %%
car_sales.dtypes

# %%
# Split into X/y
X = car_sales.drop('Price', axis=1)
y = car_sales['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# Build machine learning model
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)

# %%

# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 將非數字資料轉換為數字, 其中Doors較為特別, 雖然它已經是數字, 但是幾個車門可以作分類, 因此也加入分類
categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
transformed_x = transformer.fit_transform(X)
pd.DataFrame(transformed_x)


# %% [raw]
# # 還原
# dummies = pd.get_dummies(car_sales[['Make', 'Colour', 'Doors']])
# dummies

# %%
# 因為汽車的價格無法從顏色、廠牌、車門數來做出準確預測所以分數很低, 
# 但本節重點在於如何將廢數據化的資料轉換為數字
# model.score(X_test, y_test)

# %% [markdown]
# ### 1.2 What if where missing values)?
# 1. Fill them with some value (also known as imputation)
# 2. Remove the samples with missing data altogether.
#

# %%
# Import car sales misssing data
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing

# %%
car_sales_missing.isna().sum()

# %%
# Create X & y
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']


# %%
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# categorical_features = ['Make', 'Colour', 'Doors']
# one_hot = OneHotEncoder()
# transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
# transformed_x = transformer.fit_transform(X)
# pd.DataFrame(transformed_x)

# %% [markdown]
# ### Option 1:Fill missing data with Pandas

# %%
# Fill the 'Make' column
car_sales_missing['Make'].fillna('missing', inplace=True)

# Fill the 'Colour' column
car_sales_missing['Colour'].fillna('missing', inplace=True)

# Fill the 'Odometer (KM)' column
car_sales_missing['Odometer (KM)'].fillna(car_sales_missing['Odometer (KM)'].mean(), inplace=True)

# Fill the 'Doors' column
car_sales_missing['Doors'].fillna(4, inplace=True)


# %%
# Check our dataframe again
car_sales_missing.isna().sum()

# %%
# Remove row with missing Price value
# 因為 Price 是 label, 當 label缺失時, 將無法準確訓練model
car_sales_missing.dropna(inplace=True)



# %%
car_sales_missing.isna().sum()

# %%
len(car_sales_missing)

# %%
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']

# %%
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
transformed_x = transformer.fit_transform(car_sales_missing)
transformed_x

# %% [markdown]
# ### Option 2:Fill missing values with Scikit-Learn

# %%
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing.head()

# %%
car_sales_missing.isna().sum()

# %%
# Drop the rows with no labels
car_sales_missing.dropna(subset=['Price'], inplace=True)

# Split into X ＆ y
X = car_sales_missing.drop('Price', axis=1)
y = car_sales_missing['Price']


# %%
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

# %%
car_sales_filled = pd.DataFrame(filled_X, columns=['Make', 'Colour', 'Doors', 'Odometer (KM)'])
car_sales_filled.head()

# %%
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

# %%
# Now we've got our data as numbers and filled (no missing values)
# Let's fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# %% [markdown]
# ### Option 2: Filling missing data and transforming categorical data with Scikit-Learn
# This notebook updates the code in the "Getting Your Data Ready: Handling Missing Values in Scikit-Learn".
#
# The video shows filling and transforming the entire dataset (X) and although the techniques are correct, it's best to fill and transform training and test sets separately (as shown in the code below).

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
car_sales_missing = pd.read_csv('../data/car-sales-extended-missing-data.csv')
car_sales_missing.head()

# %%
# Check missing values
car_sales_missing.isna().sum()

# %%
# Drop the rows with no labels
car_sales_missing.dropna(subset=["Price"], inplace=True)
car_sales_missing.isna().sum()

# %% [markdown]
# Note the difference in the following cell to the videos, the data is split into train and test before any filling missing values or transformations take place.

# %%
from sklearn.model_selection import train_test_split

# Split into X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Split data into train and test
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# %%
# Check missing values
X.isna().sum()

# %% [markdown]
# Let's fill the missing values. We'll fill the training and test values separately to ensure training data stays with the training data and test data stays with the test data.
#
# Note: We use fit_transform() on the training data and transform() on the testing data. In essence, we learn the patterns in the training set and transform it via imputation (fit, then transform). Then we take those same patterns and fill the test set (transform only).

# %%
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

# %% [markdown]
# Now we've filled our missing values, let's check how many are missing from each set.

# %%
# Get our transformed data array's back into DataFrame's
car_sales_filled_train = pd.DataFrame(filled_X_train, 
                                      columns=["Make", "Colour", "Doors", "Odometer (KM)"])

car_sales_filled_test = pd.DataFrame(filled_X_test, 
                                     columns=["Make", "Colour", "Doors", "Odometer (KM)"])

# Check missing data in training set
car_sales_filled_train.isna().sum()

# %%
# Check missing data in test set
car_sales_filled_test.isna().sum()

# %%
# Check to see the original... still missing values
car_sales_missing.isna().sum()

# %% [markdown]
# Okay, no missing values but we've still got to turn our data into numbers. Let's do that using one hot encoding.
#
# Again, keeping our training and test data separate.

# %%
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

# %% [markdown]
# ### Fit a model
# Wonderful! Now we've filled and transformed our data, ensuring the training and test sets have been kept separate. Let's fit a model to the training set and evaluate it on the test set.

# %%
# Now we've transformed X, let's see if we can fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor

# Setup model
model = RandomForestRegressor()

# Make sure to use transformed (filled and one-hot encoded X data)
model.fit(transformed_X_train, y_train)
model.score(transformed_X_test, y_test)

# %% [markdown]
# ## 2. Choose the right estimator/algorithm for our problem
# * estimator/algorithm = machine learning model in scikit-learn
# * Classification - predicting whether a sample is on thing or another.
# * Regression - predicting a number
# * [scikit-learn ml map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

# %% [markdown]
# ### 2.1 Picking a mcachine learning model for a regression problem

# %%
# Import Boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
boston

# %%
boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.Series(boston['target'])

# %%
boston_df.head()

# %%
# How many data ?
len(boston_df)

# %%
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


# %% [markdown]
# How do we improve the score?
# What if Ridge wasn't working?

# %%

# %%

# %%

# %%
