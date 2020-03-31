# -*- coding: utf-8 -*-
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

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()
```

# Pandas CookBook


* [Question](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/pandas-exercises.ipynb)  
* [Answer](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/pandas-exercises-solutions.ipynb)


### Why Pands?
---
* Simple to use
* Integrated with many other data science & ML Python tools
* Helps you get your data ready for machine learning



### DataType
---


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

### Describe
---

```python
data = pd.read_csv('./source/car_sales.csv')
data.describe()
```

[.sum()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html?highlight=sum)

```python
data.sum()
```

[.mean()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html?highlight=mean)

```python
data.mean()
```

### Manipulating Data
---

```python
data = pd.read_csv('./source/car_sales.csv')
```

[sample()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html?highlight=sample#pandas.DataFrame.sample)

```python
# shuffle data rows, only change index order. 
# arg: frac= data rercentage (1= 100%, 0.5 = 50%).
data_shuffled = data.sample(frac=0.5)
data_shuffled
```

[reset_index()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html?highlight=reset_index#pandas-dataframe-reset-index)

```python
# reorder data rows. arg: drop=True
data_reorder = data_shuffled.reset_index(drop=True)
```

```python
data_reorder
```

[apply()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html?highlight=apply#pandas.DataFrame.apply)

```python
# each selected data aplly func
data['Odometer (KM)'] = data['Odometer (KM)'].apply(lambda x: x / 1.6)
```

```python
data
```

### I/O
---


##### read, (CSV, XLS, JSON, SQL, HTML ...)

```python
# 1.輸入 local端資料檔案路徑
csv_file = pd.read_csv('./source/car_sales.csv')
csv_file
```

```python
# 1.輸入 url
csv_file = pd.read_csv('./source/car_sales.csv')
csv_file_from_url = pd.read_csv(
    'https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv')
csv_file_from_url.head(1)
```

##### Write, (CSV, XLS, JSON, SQL, HTML ...)

```python
csv_file.to_json('./source/car_sales.json')
```

### Viewing & Select
---

```python
csv_file_from_url
```

[.head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html?highlight=head#pandas.DataFrame.head)

```python
# 由上而下的 .head()
csv_file_from_url.head(1)
```

[.tail()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html?highlight=tail#pandas.DataFrame.tail)

```python
# 由下而上的 .tail()
csv_file_from_url.tail(1)
```

[.hist()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html?highlight=hist)

```python
csv_file_from_url.age.hist()
```

[.plot()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html?highlight=plot#pandas.DataFrame.plot)

```python
csv_file_from_url.age.plot()
```

[.crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html?highlight=crosstab)

```python
some_date = csv_file_from_url.tail(5)
pd.crosstab(some_date.age, some_date.sex)
```

[.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html?highlight=groupby#pandas.DataFrame.groupby)

```python
csv_file_from_url.groupby(['sex']).mean()
```
