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

# %%
import pandas as pd
from jupyterthemes import jtplot
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))
jtplot.style()

# %%
# 2 main datatypes
series = pd.Series(['BMW', 'Toyota', 'Honda'])
series

# %%
# Series = 1-dimensional

# %%
colos = pd.Series(['Red', 'Blue', 'White'])
colos

# %% [markdown]
# ![](./source/anatomy_of_data_frames.png)

# %%
# DataFrame = 2-dimensional
car_data = pd.DataFrame({'Car make': series, 'Colour': colos})
car_data

# %%
# Import data
car_sales = pd.read_csv('./source/car_sales.csv')
car_sales

# %%
# Export data
# 若資料有多餘的index 欄位，要加入參數index=False 去除
car_sales.to_csv('./source/exported_car_sales.csv', index=False)

# %%
exported_car_sales = pd.read_csv('./source/exported_car_sales.csv')
exported_car_sales

# %%
# read_csv('some_url')

# %%
heart_disease = pd.read_csv(
    "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
heart_disease

# %%
# Practice area

# %%
# Series has arg (data, index)
user_names = ['Simon', 'Belinda', 'Troll']
user_hight = [170, 154, 6]
user_width = [65, 50, 0.013]
users_h = pd.Series(data=user_hight, index=user_names)
users_w = pd.Series(data=user_width, index=user_names)

# %%
# Series 引數可以dict
user = pd.Series({'name': 'Simon', 'age': 32})
user

# %%
# Series 引數也可以list,tuple
user = pd.Series(('Simon', 32))
user

# %%
# DataFrame has arg (index, data, columns)
users = pd.DataFrame({'Hight': users_h, 'Width': users_w})
users['BMI'] = users['Width'] / ((users['Hight'] / 100) ** 2)
users

# %%
# 資料轉置
users.T

# %% [markdown]
# ## Describe data

# %%
car_sales.dtypes

# %%
car_sales.columns

# %%
car_columns = car_sales.columns
car_columns

# %%
car_sales.index

# %%
car_sales.describe()

# %%
car_sales.info()

# %%
# .mean() 只會算出DataFrame 中可以計算的type的算術平均數(mean)
car_sales.mean()

# %%
car_prices = pd.Series([3000, 1500, 111250])
car_prices.mean()

# %%
car_sales.sum()

# %% [markdown]
# ## Viewing and selecting data

# %%
# top 5 row 
car_sales.head()

# %%
# bottom 5 row
car_sales.tail()

# %%
# .loc & .iloc
animals = pd.Series(['cat', 'dog', 'bird', 'panda', 'snake'], index=[0, 3, 9, 8, 3])
animals

# %%
# .iloc refers to position
animals.iloc[3]

# %%
# .loc refers to index
car_sales.loc[3]

# %%
animals.iloc[:3]

# %%
car_sales['Make']

# %%
car_sales.Colour

# %%
car_sales[car_sales['Make'] == 'Toyota']

# %%
car_sales[car_sales['Odometer (KM)'] > 100000]

# %%
pd.crosstab(car_sales['Make'], car_sales['Doors'])

# %%
# Groupby
# 將DataFrame 物件依 column groupby(分組),
# 分組後的DataFrameGroupBy object 可以再做運算
car_sales.groupby(['Make']).mean()

# %%
car_sales['Odometer (KM)'].plot()
# 如果程式碼無法產生圖片，可以做以下處理
# # %matplotlib inline
# import maatplotlib.pyplot as plt

# %%
car_sales['Odometer (KM)'].hist()

# %%
car_sales['Price'] = car_sales['Price'].str.replace('[\$\,\.]', '').astype(int)
car_sales['Price'].plot()

# %% [markdown]
# ## Manipulating Data

# %%
car_sales['Make'] = car_sales['Make'].str.lower()

# %%
car_sales

# %%
car_sales_missing = pd.read_csv('./source/car_sales_missing_data.csv')

# %%
car_sales_missing

# %%
car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean(), inplace=True)

# %%
car_sales_missing

# %%
# missing data process
missing_data_drop = pd.read_csv('./source/car_sales_missing_data.csv')
missing_data_drop.dropna(inplace=True)

# %%
missing_data_drop.to_csv('./source/car_sales_droped.csv')

# %%
# Column from series
# 由於使用 series 來產生 Column 
seats_column = pd.Series([5, 5, 5, 5, 5])

# New column called seats
car_sales['Seats'] = seats_column
car_sales


# %%
car_sales['Seats'].fillna(5, inplace=True)
car_sales

# %%
# Column from Python list
# 由於是 list 所以長度必須完全相同，不可以有missing data
fuel_economy = [7.5, 9.2, 5.0, 9.6, 8.7, 4.7, 7.6, 8.7, 3.0, 4.5]
car_sales['Fuel per 100KM'] = fuel_economy
car_sales

# %%
car_sales['Total fuel used'] = car_sales['Odometer (KM)'] / 100 * car_sales['Fuel per 100KM']

# %%
car_sales

# %%
# Create a column from a single value
car_sales['Number of wheels'] = 4
car_sales

# %%
car_sales['Passed road saftey'] = True
car_sales.dtypes

# %%
car_sales

# %%
