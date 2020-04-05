# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
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

# Math


## 名詞解釋
純量 scalar  
* Distance, speed, time, temperature, mass, length, area, volume, density, charge, pressure, energy,   
  work and power are all scalars.
* 不具方向性, 只有大小 ( magnitude / size ): 3.044, -7, 1$\frac{1}{2}$
* 通常使用細體: A, b  

向量vactor 
* Displacement, velocity, acceleration, force and momentum are all vectors.
* 具方向性、大小,
* 通常使用粗體: **A**, **b**
* $\overrightarrow{AB}$


### 三角學
![](../source/math_01_01.png)||
--- | ---
  
* `opp` = 對邊 (opposite)<br/>
* `hyp` = 斜邊 (hypotenuse)<br/>
* `adj` = 臨邊 (adjacent)<br/>
* `銳角` = $\angle$0$^\circ$ ~ $\angle$90$^\circ$<br/>   
* $\bigtriangleup$ABC與$\bigtriangleup$DEF即使角度相同, 且已知$\bigtriangleup$ABC部份邊長<br/> 
仍不能套用三角形函數來推算$\bigtriangleup$DEF的邊長

### 銳角三角形口訣: soh、cah、toa<br/> 
##### 三角函數 Trigonometric functions <br/>

* 多用來求邊長<br/>
* Parameters: 內角角度<br/>
* Return: 比值, 內角與兩邊長(opp、adj、hyo擇二)的比值關係<br/>
* Note: 回傳值(邊長比值)可與已知邊長推導出未知邊長<br/>

`正弦`<br/>
$\sin$($\angle$A) = $\frac{opp}{hyp}$<br/>
$\sin$($\angle$A) = $\cos$($\angle$B)<br/>
`餘弦`<br/>
$\cos$($\angle$A) = $\frac{adj}{hyp}$<br/>
$\cos$($\angle$A) = $\sin$($\angle$B)<br/> 
`正切`<br/> 
$\tan$($\angle$A) = $\frac{opp}{adj}$<br/>

##### 反三角函數 Inverse trigonometric functions<br/>

* 多用來求角度<br/>
* Parameters: 比值, 內角與兩邊長(opp、adj、hyo擇二)的比值<br/>
* Return: 內角角度<br/>
* Note: <br/>

`正弦` <br/> 
$\sin$$^{-1}$($\angle$A) = $\frac{opp}{hyp}$<br/> 
$\angle$A = $\sin$$^{-1}$($\frac{opp}{hyp}$)<br/>
$\angle$A = $\sin$$^{-1}$($\frac{BC}{AB}$)<br/>

`餘弦`<br/> 
$\cos$$^{-1}$($\angle$A) = $\frac{adj}{hyp}$<br/> 
$\angle$A = $\cos$$^{-1}$($\frac{adj}{hyp}$)<br/>
$\angle$A = $\cos$$^{-1}$($\frac{AC}{AB}$)<br/>
`正切`<br/> 
$\tan$$^{-1}$($\angle$A) = $\frac{opp}{adj}$<br/> 
$\angle$A = $\tan$$^{-1}$($\frac{opp}{adj}$)<br/>
$\angle$A = $\tan$$^{-1}$($\frac{BC}{AC}$)<br/>

##### 正弦定理<br/>
![](../source/math_01_04.png)||
--- | ---  
[正弦定理](https://zh.wikipedia.org/wiki/%E6%AD%A3%E5%BC%A6%E5%AE%9A%E7%90%86)  <br/>

##### 餘弦定理<br/>
* 當知道三角形的兩邊和一角時，餘弦定理可被用來計算第三邊的長，或是當知道三邊的長度時，可用來求出任何一個角。

[餘弦定理](https://zh.wikipedia.org/wiki/%E9%A4%98%E5%BC%A6%E5%AE%9A%E7%90%86)  <br/>
##### 直角三角形角度互補關係<br/>
![](../source/math_01_03.png)|![](../source/math_01_02.png)
--- | ---
  



```python
high_var_array = np.array([1, 100, 200, 300, 4000, 5000])
low_var_array = np.array([2, 4, 6, 8, 10])
```

### 指數 exponent
---
* 2$^{5}$, $^{5}$就是指數, 1 $\times$ 2 $\times$ 2 $\times$ 2 $\times$ 2 $\times$ 2 <br/>
* n$^{0}$ = 1
* n$^{1}$ = n

[np.log()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html)



### 對數 Logarithm
---

[np.log()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)



### 平方根 Sqrt
---
[np.sqrt()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)


### 算術平均數 Arithmetic mean
---
[np.mean()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)

```python
# Demo of std 
np.mean(high_var_array), np.mean(low_var_array)
```

### 方差/變異數 Variance
---
[np.var()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)  
Variance = measure of the average degree to which each number is difference to mean  
Higher variance = wider range of numbers  
Lower variance = lower range of numbers  

```python
# Demo of var
np.var(high_var_array), np.var(low_var_array)
```

<!-- #region -->


### 標準差Standard Deviation
[np.std()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
     
---
Strandard deviation = squareroot of variance  
var = std ** 0.5  

Standard deviation = a measure of how sparead out a group of numbers is from the mean
<!-- #endregion -->

```python
# Demo of std 
np.std(high_var_array), np.std(low_var_array)
```

### 內積 Dot Product


### Element Wise
