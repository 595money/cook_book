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
# ## 代數基礎

# %% [markdown]
# * inequalities (不等式)
# 若c為 正數 且 a > b；則 ac > bc。   
# 若c為 正數 且 a < b；則 ac < bc。  
# 若c為 負數 且 a > b；則 ac < bc。  
# 若c為 負數 且 a < b；則 ac > bc。  
# -2x > 20  
# x < $\frac{20}{-2}$    
# * linear equations (線性方程式/一次方程式)  
# 所有解可以在座標圖上形成一直線  
# y = 2x-3 |x|y|  
# ---|---|---
# |0|-3
# |1|-1
# |2|1
#
#
# * intercepts (截距)  
# 線性方程式於座標上畫出一直線.  
# 該線段通過X軸處為 x-intercepts, 該點座標為(n, 0)  
# 該線段通過y軸處為 y-intercepts  該點座標為(0, n)  
#
# * slope (斜率)  
# slope = $\frac{\Delta{y}}{\Delta{x}}$  
# y 軸增加的量除以 X 軸增加的量
