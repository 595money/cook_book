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

## Heart dissease project
---
This project is about claissifying whether or not a patient has heart disease.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from jupyterthemes import jtplot
jtplot.style()
```

```python
df = pd.read_csv('./source/original.csv')
```

The following table shows heart disease information for patients.

```python
df.head(10)
```

```python
df.target.value_counts().plot(kind='bar')
```

![ ](./source/original.png)


## 1.ProblemDefinition
Predicet heart disease.

```python

```

```python

```

```python

```

```python

```

## 2.Data
This is the data we're using.

```python

```

```python

```

```python

```
