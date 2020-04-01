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
import numpy as np
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()

# %% [markdown]
# ## DataTypes & Attributes

# %%
# NumPy's main datatype is ndarray
# n dimensional array
# 所有的 Data 都會轉換為數字來進行機器學習, 
# 所以需要極為複雜的維度來組織 Data (數值)
a1 = np.array([1, 2, 3])
a1

# %%
type(a1)

# %%
a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])
a3 = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]]])
a4 = np.array([[[[2, 3],
                 [2, 3],
                 [2, 3]],
                [[2, 3],
                 [2, 3],
                 [2, 3]],
                [[2, 3],
                 [2, 3],
                 [2, 3]]],
               [[[2, 3],
                 [2, 3],
                 [2, 3]],
                [[2, 3],
                 [2, 13],
                 [2, 3]],
                [[2, 3],
                 [2, 3],
                 [2, 3]]]]

              )

# %%
a2

# %%
a3.shape

# %%
a4.shape

# %%
a1.ndim, a2.ndim, a3.ndim, a4.ndim

# %%
a1.size, a2.size, a3.size, a4.size

# %%
# Create a DataFrame from a NumPy array
pd.DataFrame(a2)

# %% [markdown]
# ## 2. Creating Arrays

# %%
sample_array = np.array([1, 2, 3])
sample_array

# %%
sample_array.dtype

# %%
ones = np.ones([1, 2])
ones

# %%
zeros = np.zeros([1, 2])
zeros

# %%
range_array = np.arange(0, 10, 2)
range_array

# %%
random_array = np.random.randint(1, 10, size=(3, 3, 3))
random_array

# %%
random_array.size

# %%
random_array.shape

# %%
random_array2 = np.random.random((5, 3))
random_array2

# %%
random_array_3 = np.random.rand(5, 3)
random_array_3

# %%
# Pseudo-random number
# 隨機的只有種子編碼, 所以種子編碼固定後, 得到的亂數都會相同
np.random.seed(1)
random_array_4 = np.random.randint(10, size=(5, 3))
random_array_4

# %% [markdown]
# ## 3. Viewing arrays and 

# %%
np.unique(random_array_4)

# %%
random_array

# %%
random_array[:2, :2, :2]

# %%
a5 = np.random.randint(10, size=(2, 3, 4, 5))
a5.shape, a5.ndim

# %%
# Get the first 4 numbers of the inner most arrays
a5

# %%
a5[:1, :2, :3, :4]

# %% [markdown]
# ## 4. Manipulating & comparing arrays

# %% [markdown]
# ### Arithmetic

# %%
a1

# %%
ones = np.ones(3)
ones

# %%
a1 + ones

# %%
a2

# %%
a3

# %%
# 同一維度必須相等或其中一方為1
a3[:, :2, :1]

# %%
a3.reshape(3, 2, 3)
a3

# %%
a1 / ones

# %%
# Floor division removes the decimals (rounds down)
a2 / a1

# %%
a2 // a1

# %%
a2 ** 2

# %%
np.square(a2)

# %%
a1 + ones

# %%
np.add(a1, ones)

# %%
a2

# %%
a2 % 2

# %%
np.log(a1)

# %% [markdown]
# ## Aggregation
# Aggregation = performing the same operation on a number of things

# %%
listy_list = [1, 2, 3]
type(listy_list)

# %%
sum(listy_list)

# %%
a1

# %%
type(a1)

# %%
np.sum(a1)

# %% [markdown]
# User Python's method( `sum()` ) on Python datatypes and use NumPy's methods on NumPy arrays( `np.sum()` )

# %%
# Creative a massive NumPy array
massive_array = np.random.random(100000)
massive_array.size
massive_array[:10]

# %%
# %timeit sum(massive_array) # Python's sum() 
# %timeit np.sum(massive_array) # NumPy's sum()

# %%
a2

# %%
np.mean(a2)

# %%
np.max(a2) ,np.min(a2), np.std(a2)

# %%
