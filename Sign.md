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

# 概述
* [LaTex mathematical symbols](
https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)


### 希臘字母
Sign <img width=200/>|Code<img width=400/>|Discript
:---|---|--
$\alpha$ |`$\alpha$`|alpah
$\beta$ |`$\beta$`|beta
$\gamma$ |`$\gamma$`|gamma
$\Gamma$ |`$\Gamma$`|Gamma
$\delta$|`$\delta$`|delta
$\Delta$|`$\Delta$`|Delta
$\epsilon$|`$\epsilon$`|epsilon
$\varepsilon$|`$\varepsilon$`|varepsilon
$\zeta$|`$\zeta$`|zeta
$\eta$|`$\eta$`|eta
$\theta$|`$\theta$`|theta
$\Theta$|`$\Theta$`|Theta
$\vartheta$|`$\vartheta$`|vartheta
$\iota$|`$\iota$`|iota
$\pi$|`$\pi$`|pi
$\phi$|`$\phi$`|phi
$\psi$|`$\psi$`|psi
$\Psi$|`$\Psi$`|Psi
$\omega$|`$\omega$`|omega
$\Omega$|`$\Omega$`|Omega
$\chi$|`$\chi$`|chi
$\rho$|`$\rho$`|rho
$\omicron$|`$\omicron$`|omicron
$\sigma$|`$\sigma$`|sigma
$\Sigma$|`$\Sigma$`|Sigma
$\nu$|`$\nu$`|nu
$\xi$|`$\xi$`|xi
$\tau$|`$\tau$`|tau
$\lambda$|`$\lambda$`|lambda
$\Lambda$|`$\$Lambda`|Lambda
$\mu$|`$\mu$`|mu
$\partial$|`$\partial$`|partial




### 數學符號

 Sign <img width=200/>|Code<img width=400/>|Discript
:---|---|--
$\sum$ |`$\sum$`|求和公式
$\sum_{i=123}^n$|`$\sum_{i=123}^n$` |求和上下標
$y=\begin{cases}x\\\alpha\end{cases}$|`$y=\begin{cases}x\\\alpha\end{cases}$`|大括號
$\times$|`$\times$`|乘號
$\div$|`$\div$`|除號
$\mid$|`$\mid$`|
$\pm$|`$\pm$`|正負號
$\cdot$|`$\cdot$`|點
$\circ$|`$\circ$`|圈
$\ast$|`$\ast$`|星號
$\bigotimes$|`$\bigotimes$`|克羅內克積
$\leq$|`$\leq$`|小於等於
$\geq$|`$\geq$`|大於等於
$\neq$|`$\neq$`|不等於
$\not>$|`$\not>$`|不大於
$\approx$|`$\approx$`|約等於
$\equiv$|`$\equiv$`|約等於
$\prod$|`$\prod$`|N元乘積
$\coprod$|`$\coprod$`|N元餘積
$\cdots$|`$\cdots$`|省略號
$\because$|`$\because$`|因為
$\therefore$|`$\therefore$`|所以
$\forall$|`$\forall$`|任意
$\exists$|`$\exists$`|存在
9$\frac{1}{2}$|`9$\frac{1}{2}$`|分號
1$^{999}$|`1$^{999}$`|次方














### 對數符號
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\log$|`$\log$`|
$\lg$|`$\lg$`|
$\ln$|`$\ln$`|


### 集合符號
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\emptyset$|`$\emptyset$`|空集合
$\in$|`$\in$`|屬於
$\not\subset$|`$\notin$`|不屬於
$\subset$|`$\subset$`|子集合
$\supset$|`$\supset$`|子集合
$\subseteq$|`$\subseteq$`|真集合
$\supseteq$|`$\supseteq$`|真集合
$\bigcup$|`$\bigcup$`|聯集
$\bigcap$|`$\bigcap$`|交集
＼|`\`|差集
$\bigwedge$|`$\bigwedge$`|邏輯與
$\bigvee$|`$\bigvee$`|邏輯或
$\bigoplus$|`$\bigoplus$`|互斥或
$\biguplus$|`$\biguplus$`|多重集
$\bigsqcup$|`$\bigsqcup$`|互斥聯集


### 微積分
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\prime$|`$\prime$`|
$\int$|`$\int$`|
$\iint$|`$\iint$`|
$\iiint$|`$\iiint$`|
$\iiiint$|`$\iiiint$`|
$\oint$|`$\oint$`|
$\lim$|`$\lim$`|極限
$\infty$|`$\infty$`|無窮
$\nabla$|`$\nabla$`|梯度




### 箭頭符號
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\uparrow$|`$\uparrow$`|
$\downarrow$|`$\downarrow$`|
$\Uparrow$|`$\Uparrow$`|
$\Downarrow$|`$\Downarrow$`|
$\rightarrow$|`$\rightarrow$`|
$\leftarrow$|`$\leftarrow$`|
$\Rightarrow$|`$\Rightarrow$`|
$\Leftarrow$|`$\Leftarrow$`|
$\longrightarrow$|`$\longrightarrow$`|
$\longleftarrow$|`$\longleftarrow$`|
$\Longrightarrow$|`$\Longrightarrow$`|
$\Longleftarrow$|`$\Longleftarrow$`|









### 三角運算
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\bigtriangleup$|`$\bigtriangleup$`|三角形
$\bigtriangledown$|`$\bigtriangledown$`|倒三角形
$\bot$|`$\bot$`|
$\angle$|`$\angle$`|角度
$\measuredangle$|`$\measuredangle$`|被測量的角度 
30$^\circ$|`30$^\circ$`|角度表示
$\sin$|`$\sin$`|
$\cos$|`$\cos$`|
$\tan$|`$\tan$`|
$\cot$|`$\cot$`|
$\sec$|`$\sec$`|
$\csc$|`$\csc$`|






### 連線符號
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\overline{a b c d}$|`$\overline{a b c d}$`|
$\underline{a b c d}$|`$\underline{a b c d}$`|
$\overbrace{a \underbrace{b c}_{1.0} d}^{2.0}$|`$\overbrace{a \underbrace{b c}_{1.0} d}^{2.0}$`|
$X\vec{x}_{\vec{x}_{\vec{x}}}$ $\vec{a}\vec{b}\vec{m}\vec{X}$|`$X\vec{x}_{\vec{x}_{\vec{x}}}$ $\vec{a}\vec{b}\vec{m}\vec{X}$`|
$\overleftarrow{blahblahblah}$|`$\overleftarrow{blahblahblah}$`|
$\overrightarrow{blahblahblah}$|`$\overrightarrow{blahblahblah}$`|


### 音符？
Sign <img width=200/>|  Code<img width=400/>|Discript
:---|---|--
$\hat{y}$|`$\hat{y}$`|期望值
$\breve{y}$|`$\breve{y}$`|短音符
$\check{y}$|`$\check{y}$`|抑揚符
