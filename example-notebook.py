# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
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
# ## Heart dissease project
# ---
# This project is about claissifying whether or not a patient has heart disease.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from jupyterthemes import jtplot
jtplot.style()

# %%
df = pd.read_csv('./source/original.csv')

# %% [markdown]
# The following table shows heart disease information for patients.

# %%
df.head(10)

# %%
df.target.value_counts().plot(kind='bar')

# %% [markdown]
# ![ ](./source/original.png)

# %% [markdown]
# ## 1.ProblemDefinition
# Predicet heart disease.

# %%

# %%

# %%

# %%

# %% [markdown]
# ## 2.Data
# This is the data we're using.

# %%

# %%

# %%
