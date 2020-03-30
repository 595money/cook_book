---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:70% !important; }</style>"))
jtplot.style()
```

# Pandas CookBook
---


### DataType


##### Series, 1維資料結構

```python
user_name = pd.Series(['Simon', 'Jay', 'Leo'], name='user_name')
user_name
```

```python
# 顯示出此資料結構的 (Rows, Columns)
# Series 為一維資料, 所以只有 Rows有值, Colums 不帶值
user_name.shape
```

```python
user_name.describe
```

##### DataFrame, 2維資料結構, 每一個 cloumn 可以切割為一個 Sseries

```python
user_age = pd.Series([32, 35, 11], name='user_age')
users = pd.DataFrame({user_name.name: user_name, user_age.name: user_age})
users
```

```python
users.shape
```

```python
users.info()
```

```python

```

```python

```

### I/O


##### read, (CSV, XLS, JSON, SQL, HTML ...)

```python
# 1.輸入 local端資料檔案路徑
csv_file = pd.read_csv('./source/car_sales.csv')
csv_file
```

```python
# 1.輸入 url
csv_file = pd.read_csv('./source/car_sales.csv')
csv_file_from_url = pd.read_csv('https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv')
csv_file_from_url.head(1)
```

##### Write, (CSV, XLS, JSON, SQL, HTML ...)

```python
csv_file.to_json('./source/car_sales.json')
```

##### select method : .head(), .tail()

```python
csv_file_from_url
```

```python
# 由上而下的 .head()
csv_file_from_url.head(1)
```

```python
# 由下而上的 .tail()
csv_file_from_url.tail(1)
```
