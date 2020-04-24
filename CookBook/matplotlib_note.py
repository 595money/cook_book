# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
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

# %%
import pandas as pd
import numpy as np
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()

# %% [markdown]
# # matplotlib CookBook

# %% [markdown]
# * [Question](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/matplotlib-exercises.ipynb)  
# * [Answer](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/matplotlib-exercises-solutions.ipynb)

# %% [markdown]
# ### Why matplotlib?
# ---
# * Built on NumPy arrays (and Python)
# * Integrates directly with pandas
# * Can create basic or advanced plots
# * Simple to use interface (once you get the foundations)

# %% [markdown]
# ![](../source/matplotlib_01.png)
# * Importing Matplotlib and the 2 ways of plotting
# * Plotting data from NumPy arrays
# * Plotting data from pandas DataFrames
# * Savihng and sharing plots
