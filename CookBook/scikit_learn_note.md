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
  0. An end-to-end Scikit-Learn workflow
  1. Getting the data ready
  2. Choose the right estimator/algorithm for our problem　　
  3. Fit th model/algorithm and use it to make predictions on our data
  4. Evaluating a model
  5. Improve a model
  6. Save and load a trained model
  7. Putting it all together
  

```python
import pandas as pd
import numpy as np
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

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
