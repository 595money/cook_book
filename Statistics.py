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
# 0. import matplotlib and get it ready for plotting in Jupyter
# %matplotlib inline
import matplotlib.pyplot as plt

# %% [markdown]
# # Statistics

# %% [markdown]
# ### 高斯分布 Gaussian distribution/normal distribution
# ---
# [np.random.randn(1000)](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html)

# %%
# Make some data
# 高斯分布、標準常態分布
x = np.random.randn(1000)
fig, ax = plt.subplots()
ax.hist(x);
