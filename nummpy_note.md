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
import pandas as pd
import numpy as np
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()
```

# NumPy CookBook


* [Question](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/numpy-exercises.ipynb)  
* [Answer](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/numpy-exercises-solutions.ipynb)


### Why NumPy?
---
* It's fast (C language)
* Behind the scenes optimizations written in C
* Vectorization via broadcasting (avoiding loops)
* Backbone of other Python scientific pakcages


### What are we going to cover?
---
* Most userful functions
* NumPy datatypes & attributes (ndarray)
* Creating arrays
* Viewing arrays & matrices
* Sorting arrays
* Use cases
