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

# %%
# %matplotlib inline
import matplotlib.pylab as plt
import pandas as pd
from jupyterthemes import jtplot
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))
jtplot.style()

# %% [markdown]
# # Introduction to Matplotlib

# %%
plt.plot()
plt.show()

# %%
plt.plot([1, 2, 3, 4])

# %%
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]
plt.plot(x, y)

# %%
# 1st method
fig = plt.figure() # creates a figure
ax = fig.add_subplot() # adds some axes
plt.show()

# %%
# 2nd method
fig = plt.figure() # creates a figure
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y) # add some data
plt.show()

# %%
# 3rd method (recommended)
fig, ax = plt.subplots()
plt.plot(x, [50, 100, 200, 250]);
