---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
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
%matplotlib inline
import matplotlib.pylab as plt
import pandas as pd
from jupyterthemes import jtplot
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))
jtplot.style()
```

# Introduction to Matplotlib

```python
plt.plot()
plt.show()
```

```python
plt.plot([1, 2, 3, 4])
```

```python
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]
plt.plot(x, y)
```

```python
# 1st method
fig = plt.figure() # creates a figure
ax = fig.add_subplot() # adds some axes
plt.show()
```

```python
# 2nd method
fig = plt.figure() # creates a figure
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y) # add some data
plt.show()
```

```python
# 3rd method (recommended)
fig, ax = plt.subplots()
plt.plot(x, [50, 100, 200, 250]);
```
