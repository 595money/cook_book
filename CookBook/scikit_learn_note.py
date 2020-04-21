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
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)

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

