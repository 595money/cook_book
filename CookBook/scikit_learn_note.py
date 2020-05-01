# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
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
# 1. Get the data ready
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
# 將data處理好已進入機器學習  
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
# 根據要處理的問題選擇使用的機器學習模型
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
# Let's refer bsck to the map ... [see also](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

# %%
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

# %%
# Check the ridge model again
model.score(X_test, y_test)

# %% [markdown]
# ### 2.2 Choosing and estimator for a classification problem
# Let's go to the map...[see also](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

# %%
heart_disease = pd.read_csv('../data/heart-disease.csv')
len(heart_disease)

# %% [markdown]
# Consulting the map and it says to try `LinearSVC`.

# %%
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


# %% [markdown]
# Tidbit:
#     1. If you have structured data, used ensemble methods
#     2. If you have unstructured data, use deep learning or transfer learning

# %% [markdown]
# ## 3. Fit the model/algorithm and use it to make predictions on our data  
#
#
# ### 3.1 Fitting the model to the data
#
# Different names for:
# * X = features, features variables, data
# * y = labels, targets, target variables

# %%
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

# %%
X.head()

# %% [markdown]
# ### 3.2 Make predictions using a machine learning model
#
# 2 way to make predictions:
# 1. `predict()`
# 2. `predict_proba()`

# %%
# Use a trained model to make predictions

# %%
clf.predict(X_test)

# %%
np.array(y_test)

# %%
# Compare predictions to truth labels to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)

# %%
clf.score(X_test, y_test)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_preds)

# %% [markdown]
# Make predictions with predict_proba()
#

# %%
# predict_proba() returns probabilities of a calssification label
clf.predict_proba(X_test[:5])

# %%
# Let's predict() on the same data...
clf.predict(X_test[:5])

# %% [markdown]
# `predict()` can also be used for regression models.

# %%
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


# %%
y_preds[:10]

# %%
# y_test is Answer
np.array(y_test[:10])

# %%
# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error
from typing import List

# 平均絕對誤差 mean absolute error
mean_absolute_error(y_test, y_preds)

# %% [markdown]
# ### 4. Evaluating a machine learning modle
#
# Three ways to evaluate Scikit-Learn models/esitmators:  
# 評估機器學習模型
# 1. Estimator `score` method 
# 2. The `scoring` parameter
# 3. Problem-specific metric functions.  
#
# ### 4.1 Evaluation a model with the score method

# %%
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %% [markdown]
# Let's do the same but for regreesion...

# %%
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# Create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model
model = RandomForestRegressor().fit(X_train, y_train)

# %%
model.score(X_test, y_test)

# %% [markdown]
# ### 4.2 Evaluating a model using the `scoring` parameter

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%
# clf = train model
# X = feature
# y = label
cross_val_score(clf, X, y)

# %%
np.random.seed(42)
# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y))

# Compare the two
clf_single_score, clf_cross_val_score

# %%
# Scoring parameter set to None by default
# 當cross_val_score參數位給scoring時, 預設為None,
# 而 scoring = None (預設), 則會帶入 scoring = clf.score()
# 此處的 clf 是 RandomForestClassifier, 所以預設scoring 是 mean accuracy
# Default scoring parameter of classifier = mean accuracy
# clf.score() = mean accuracy

cross_val_score(clf, X, y, cv=5, scoring=None)

# %% [markdown]
# ### 4.2.1 Classification model evaluation metrics
# 1. Accuracy
# 2. Area umder ROC curve
# 3. Confusion matrix
# 4. Classification report
#
# **Accuracy**

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

clf = RandomForestClassifier()
cross_val_score = cross_val_score(clf, X, y)

# %%
np.mean(cross_val_score)

# %%
print(f'Heart Disease Classifier Cross-Validated Accuracy:{np.mean(cross_val_score) * 100:.2f}%')

# %% [markdown]
# **Area under the receiver operating characteristic curve (AUC/ROC)**
# * Area under curve (AUC)
# * ROC curve
# * [see also](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
#
# ROC curves are a comparison of a model's true postive rate (tpr) versus a models false positive rate (fpr).
#
# * True positive = model predicts 1 when truth is 1 
# * False positive = model predicts 1 when truth is 0  
# * True negative = model predicts 0 when truth is 0
# * Flase negative = model predicts 0 when true is 1

# %%
# Create X_test... etc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from sklearn.metrics import roc_curve

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:10], len(y_probs)

# %%
y_probs_positive = y_probs[:, 1]
y_probs_positive[:10]

# %%
# Caculate fpr, tpr and thresholds
# fpr = FP / (FP + TN) 在所有實際陰性樣本中被誤判為陽性的比率, T = 猜對, P = 實際是陽性
# tpr = TP / (TP + FN) 在所有實際陽性樣本中正確判定陽性的比率, F = 猜錯, P = 實際是陽性
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

# Check the false positive rates
fpr

# %%
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

# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_probs_positive)

# %%
# Plot perfect ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve(fpr, tpr)

# %%
# Perfect AUC score
roc_auc_score(y_test, y_test)

# %% [markdown]
# **Confusion Matrix**  
# A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.  
# 混淆矩陣是比較模型預測的標籤和模型應預測的實際標籤的快速方法。
#   
# In essence, giving you an idea of where the model is getting confused.  
# 從本質上講，讓您了解模型的混亂之處

# %%
from sklearn.metrics import confusion_matrix

# y_test = test sets 的 label
# y_preds = 使用 X_test 進行預測得到的 y_preds
y_preds = clf.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
# Visualize confusion matrix with pd.crosstab()
pd.crosstab(y_test,
            y_preds,
            rownames=['Actual Labels'],
            colnames=['Predicted Labels'])

# %%
# install seaborn
# How install a conda package into the current envrionment from a Jupyter Notebook
# import sys
# # !conda install --yes --prefix {sys.prefix} seaborn

# %%

# %%
# Make our confusion matrix more visual with Seaborn's heatmp()
import seaborn as sns

# Set the font scale
sns.set(font_scale=1.5)

# Create a confusion martix
conf_mat = confusion_matrix(y_test, y_preds)

# Plot it using Seaborn
sns.heatmap(conf_mat);


# %%
def plot_conf_mat(conf_mat):
    '''
    Plots a confusion matrix ustin Seaborn's heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(conf_mat,
                     annot=True,  # Annotate the boxes with conf_mat info
                     cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label');


plot_conf_mat(conf_mat)


# %%
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, X_test, y_test);

# %% [markdown]
# **Classification report**

# %%
from sklearn.metrics import classification_report

# y_test = test sets 的 label
# y_preds = 使用 X_test 進行預測得到的 y_preds
# 將y_test 與 y_preds進行比較
print(classification_report(y_test, y_preds))


# %% [markdown]
# 1:41 記得重看
# Precision = TP/ (TP + FP)   
# Recall = TP/(TP + FN)  
# F1-score = 2* (P*R)/(P+R)  
# Suppot = 樣本數  
# Accuracy   
# Macro avg  
# Weighter abg   

# %%
# Where precision and recall become valuable
disease_true = np.zeros(10000)
disease_true[0] = 1  # only one positive case

disease_preds = np.zeros(10000)  # model predicts every case as 0

pd.DataFrame(classification_report(disease_true,
                                   disease_preds,
                                   output_dict=True))

# %% [markdown]
# To summarize classification metrics:
#  * **Accuracy** is a good measure to start with if all classes are balanced (e.g. same amount of samples which are labelled with 0 or 1).  
#  ##### 如果類別間的樣本數均達到平衡（例如，標有0或1的相同數量的樣本）可以安心使用
#  * **Precision** and **recall** become more important when classes are imbalanced.
#  ##### 如果類別間的樣本書不平衡時, 精確度與召回率變得更加重要
#  * if false positive predictions are worse than false negatives, aim for higher precision.  
#  ##### 如果FP比FN差, 提高精確度的加權
#  * if false negative predictions are worse than false positives, aim for higher recall.  
#  ##### 如果FN比FP差, 提高召回率的加權
#  * **F1-score** is a combination of precision and recall.

# %% [markdown]
# ### 4.2.2 Regression model evaluation metrics
#
# Model evaluation metrics documentation - [see also](https://scikit-learn.org/stable/modules/model_evaluation.html) [cn](https://sklearn.apachecn.org/docs/master/32.html)
#
# 1. R^2 (pronounced r-squared) or coefficient of determination.
# 2. Mean absolute error (MAE)
# 3. Mean suqared error (MSE)
# 4. perfectly is max(R^2) & min(MAE) & min(MSE)  
#   
# **R^2**  
# What R-squared does: Compares your model predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. Forexample, if all your model does is predict the mean of the targets, it's R^2 value woule be 0. And if your model perfectly predicts a range of numbers it's R^2 value would be 1.
#
# ##### R平方的作用：
# #### 將模型預測與目標均值進行比較。值的範圍可以從負無窮大（非常差的模型）到1。 
# ##### 例如，如果您的所有模型所做的都是預測目標的均值，則R ^ 2值將為0。
# ##### 並且，如果您的模型完美地預測了數字範圍，則為R ^ 2的值為1。
#
# 描述出數值怎樣算好?
# 接近0與接近１代表甚麼意思？

# %%
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

X = boston_df.drop('target', axis=1)
y = boston_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# %%
# default R^2
model.score(X_test, y_test)

# %%
from sklearn.metrics import r2_score

# Fill an array with y_test mean
y_test_mean = np.full(len(y_test), y_test.mean())
y_test_mean

# %%
# Model only predicting the mean gets an R^2 score of 0
r2_score(y_test, y_test_mean)

# %%
# Model predicting perfectly the correct values gets an R^2 scroe of 1
r2_score(y_test, y_test)

# %% [markdown]
# **Mean absolue error(MAE)**  
# MAE is the average of the aboslute differences between predictions and actual values.  
# ##### MAE 為預測值與實際值的差異絕對值的平均,
# It gives you an idea of how wrong your models predictions are.
# ##### 可以用來了解model 預測的錯誤程度  

# %%
# Mean absolute error
y_preds = model.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)
df = pd.DataFrame(data={'actual values': y_test,
                        'predicted values': y_preds})
df['differences'] = abs(df['predicted values'] - df['actual values'])
df

# %% [markdown]
# **Mean Squared error (MSE)**

# %%
# Mean Squared error
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_preds)
mse

# %%
squared = np.square(df['differences'])
squared.mean()

# %% [markdown]
# ### 4.2.3 Finally using the `scoring` parameter

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
clf = RandomForestClassifier()

# %%
np.random.seed(42)
cv_acc = cross_val_score(clf, X, y, scoring=None)
# Cross-validated accuracy
print(f'The cross-validated accuracy is: {np.mean(cv_acc) * 100 :.2f}%')

# %%
np.random.seed(42)
cv_acc = cross_val_score(clf, X, y, scoring='accuracy')
# Cross-validated accuracy
print(f'The cross-validated accuracy is: {np.mean(cv_acc) * 100 :.2f}%')

# %%
# Precision
np.random.seed(42)
cv_precision = cross_val_score(clf, X, y, scoring='precision')
np.mean(cv_precision)

# %%
# Recall
cv_recall = cross_val_score(clf, X, y, scoring='recall')
np.mean(cv_recall)

# %%
# F1
cv_f1 = cross_val_score(clf, X, y, scoring='f1')
np.mean(cv_f1)

# %% [markdown]
# How about out regression model?

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = boston_df.drop('target', axis=1)
y = boston_df['target']
model = RandomForestRegressor()

# %%
# R^2
np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring=None)
np.mean(cv_r2)

# %%
# R^2
np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
np.mean(cv_r2)

# %%
# Mean absolute error (MAE)
# 遵循一貫作風高分比低分好, 所以MAE與MSE等都改為return negtive score
np.random.seed(42)
cv_mae = cross_val_score(model, X, y, scoring='neg_mean_absolute_error')
np.mean(cv_mae)

# %%
# Mean squared error (MSE)
np.random.seed(42)
cv_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
np.mean(cv_mse)

# %% [markdown]
# ### 4.3 Using different evaluation metrics as Scikit-Learn functions  
# **Classification evaluation functions**
#

# %%

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make some predictions
y_preds = clf.predict(X_test)

# Evaluate the classifier
print('Classifier metrics on the test set')
print(f'Accuracy: {accuracy_score(y_test, y_preds) * 100:.2f}%')
print(f'Precision: {precision_score(y_test, y_preds) * 100:.2f}%')
print(f'Recall: {recall_score(y_test, y_preds) * 100:.2f}%')
print(f'F1: {f1_score(y_test, y_preds) * 100:.2f}%')

# %% [markdown]
# **Regression evaluation functions**

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = boston_df.drop('target', axis=1)
y = boston_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions using our regression model
y_preds = model.predict(X_test)

# Evaluate the regression model

print('Regression metrics on the test set')
print(f'R^2: {r2_score(y_test, y_preds) * 100:.2f}%')
print(f'MAE: {mean_absolute_error(y_test, y_preds)}')
print(f'MSE: {mean_squared_error(y_test, y_preds)}')

# %% [markdown]
# ## 5. Improving a model
# 由於做了初次預測後, 會開始對預測結果進行優化,
# 所以初次預測的成果也會是基準.
#
# First predictions = baseline prediction.  
# First model = baseline model.  
#
# From a data perspective:
# * Could we collect more data ? (generally, the more, the better)
# * Could we improve our data?
#
# From a model perspective:
# * Is there a better model we could use?
# * Could we improve the current model?
#
# Hyperparameters vs. Parameters
# * Parameters = model find theese patterns in data
# * Hyperparameters = settings on a model you can adjust to (potentially) improve it's ability to find patterns

# %% [markdown]
# Three ways to adjust hyperparameters:
# 1. By hand
# 2. Randomly with RandomSearchCV
# 3. Exhaustively with GridSearchCV

# %%
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.get_params()


# %% [markdown]
# ### 5.1 Tuning hyperparameters by hand
# Let's make 3 sets, training, validation and test.

# %% [markdown]
# We're going to try and adjust:
# * `max_depth`
# * `max_features`
# * `min_simples_leaf`
# * `min_samples_split`
# * `n_estimators`

# %%
def evaluate_preds(y_true, y_preds):
    '''
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    '''
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {'accuracy': round(accuracy, 2),
                   'precision': round(precision, 2),
                   'recall': round(recall, 2),
                   'f1': round(f1, 2)}
    print(f'Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')
    return metric_dict


# %%
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X & y
X = heart_disease_shuffled.drop('target', axis=1)
y = heart_disease_shuffled['target']

# Split the data into train, validation & test sets
train_split = round(0.7 * len(heart_disease_shuffled))  # 70% of data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled))  # 15% of data
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

clf = RandomForestClassifier()

# %%
# 查看初始化的預設參數
clf.get_params()

# %%
np.random.seed(42)
clf.fit(X_train, y_train)
# Make baseline predictions
y_preds = clf.predict(X_valid)

# Evaluate the classifier on validation set
baseline_metrics = evaluate_preds(y_valid, y_preds)

# %% [markdown]
# ### 5.2 Hyperparameter tuning with RandomizedSearchCV
# 會將grid中的所有組合做隨機嘗試

# %%
from sklearn.model_selection import RandomizedSearchCV

grid = {'n_estimators': [10, 100, 200, 500, 1000, 1200],
       'max_depth': [None, 5, 10, 20, 30],
       'max_features': ['auto', 'sqrt'],
       'min_samples_split': [2, 4, 6],
       'min_samples_leaf': [1, 2, 4]}
np.random.seed(42)
# Split into X & y
X = heart_disease_shuffled.drop('target', axis=1)
y = heart_disease_shuffled['target']

train_split = round(0.7 * len(heart_disease_shuffled))
valid_split = round(0.15 * len(heart_disease_shuffled) + train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:train_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=1, # number of models to try
                            cv=2,
                            verbose=2,
                            random_state=42, # set random_state to 42 for reproducibility
                            refit=True) # set refit=True (default) to refit the best model on the full dataset 
# 隨機測試次數為 n_iter * cv
# n_iter 迴圈次數
# cv 資料分割為幾分
# Fit the RandomizedSearchVC version of clf
rs_clf.fit(X_train, y_train)

# %%
rs_clf.best_params_

# %%
# Make predictions with the best hyperparameters
rs_y_preds = rs_clf.predict(X_test)

# Evaluate the predictions
rs_metrics = evaluate_preds(y_test, rs_y_preds)

# %% [markdown]
# ### 5.3 Hyperparameter tuning with GridSearchCV
# 會將grid中的所有組合都做嘗試

# %%
grid

# %%
6*5*2*3*3

# %%
grid_2 = {'n_estimators': [100, 200, 500],
         'max_depth': [None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_split': [6],
         'min_samples_leaf': [1, 2]}

# %%
from sklearn.model_selection import GridSearchCV, train_test_split
np.random.seed(42)
# Split into X & y
X = heart_disease_shuffled.drop('target', axis=1)
y = heart_disease_shuffled['target']

train_split = round(0.7 * len(heart_disease_shuffled))
valid_split = round(0.15 * len(heart_disease_shuffled) + train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:train_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup GridSearchCV
gs_clf = GridSearchCV(estimator=clf,
                      param_grid=grid_2,
                      cv=2,
                      verbose=2,
                      refit=True
                     ) # set refit=True (default) to refit the best model on the full dataset
                     
# 測試次數為 param_grid 所有排列組合 * cv

# Fit the RandomizedSearchVC version of clf
gs_clf.fit(X_train, y_train)

# %%
gs_clf.best_params_

# %%
gs_y_preds = gs_clf.predict(X_test)

# evaluate the predictions
gs_metrics = evaluate_preds(y_test, gs_y_preds)

# %%
compare_metrics = pd.DataFrame({'baseline': baseline_metrics,
                              'random search': rs_metrics,
                               'grid search': gs_metrics})
compare_metrics.plot.bar(figsize=(10, 8))
compare_metrics

# %% [markdown]
# ### 6. Saving and loading trained machine learning models
# Two ways to sava and load machine learning models:
# 1. With Python's `pickle` module
# 2. With the `joblib` module    
#   
#
# **Pickle**
#

# %%
import pickle

# Save an extisting model to file
pickle.dump(gs_clf, open('gs_random_random_forest_codel_1.pkl', 'wb'))

# Load a saved model 
loaded_pickle_model = pickle.load(open('gs_random_random_forest_codel_1.pkl', 'rb'))

# Make some predictions
pickle_y_preds = loaded_pickle_model.predict(X_test)
evaluate_preds(y_test, pickle_y_preds)



# %% [markdown]
# **Joblib**

# %%
from joblib import dump, load

# Save model to file
dump(gs_clf, filename='gs_random_forest_model_1.joblib')

# %%
# Import a saved joblib model
loaded_joblib_model = load(filename='gs_random_forest_model_1.joblib')

# %%
# Make and evaluate joblib predictions
joblib_y_preds = loaded_joblib_model.predict(X_test)
evaluate_preds(y_test, joblib_y_preds)

# %% [markdown]
# ### 7.Putting it all together!
# **Step we want to do (all in one all)**
# 1. Fill missing data
# 2. Convert data to numbers
# 3. Build a model on the data

# %%
data = pd.read_csv('../source/car-sales-extended-missing-data.csv')
data

# %%
data.dtypes

# %%
data.isna().sum()

# %%
# Getting data ready
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Mpdelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Setup random seed
import numpy as np 
np.random.seed(42)

# Import data and drop rows with missing labels (沒有label就無法訓練所以drop)
data = pd.read_csv('../source/car-sales-extended-missing-data.csv')
data.dropna(subset=['Price'], inplace=True)

# Define different features and transformer pipeline
categorical_features = ['Make', 'Colour']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

door_feature = ['Doors']
door_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=4))
])
numeric_features = ['Odometer (KM)']
numertic_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  
])
    
# Setup preprocessing steps (fill missing values, then convert to number)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('door', door_transformer, door_feature),
        ('num', numertic_transformer, numeric_features)
    ])

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', RandomForestRegressor())])

# Split data
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and score the model
model.fit(X_train, y_train)
model.score(X_test, y_test)

# %% [markdown]
# It's also possible to use `GridSearchCV` or `RandomizedSesrchCV` with our `Pipeline`.

# %%
# Use GridSearchCV with our regression Pipeline
pipe_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'], # preprocessor.num.imputer.strategy
    'model__n_estimators': [100, 1000],
    'model__max_depth': [None, 5],
    'model__max_features': ['auto'],
    'model__min_samples_split': [2, 4]
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)

# %%
gs_model.fit(X_train, y_train)
