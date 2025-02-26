# cs231n

CV入门

# 环境

* google drive + colab

# Assignments

## Assignment 1

> Image Classification, kNN, SVM, Softmax, Fully-Connected Neural Network

[Q1: k-Nearest Neighbor classifier](./assignments/assignment1/knn.ipynb)

[Q2: Training a Support Vector Machine](./assignments/assignment1/svm.ipynb)

[Q3: Implement a Softmax classifier](./assignments/assignment1/softmax.ipynb)

[Q4: Two-Layer Neural Network](./assignments/assignment1/two_layer_net.ipynb)

[Q5: Higher Level Representations: Image Features](./assignments/assignment1/features.ipynb)

## Assignment 2

> Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets, Network Visualization

[Q1: Multi-Layer Fully Connected Neural Networks](./assignments/assignment2/FullyConnectedNets.ipynb)

# Notes

这里放一些笔记。主要是矩阵代数相关的内容。

下面会涉及到大量的**标量对向量求导**, **向量对向量求导**, **矩阵对矩阵求导** 等内容。

推荐文章: [求导布局](https://zhuanlan.zhihu.com/p/263777564), [cs231n-linear-backprop](https://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf)

## 损失函数对权重矩阵求导

简单推导一下`SVM`和`Softmax`的损失函数对权重矩阵`W`的梯度。

首先明确一些符号。

一张图像一共有 $D$ 个像素，可以被分类为 $C$ 个类别，一共有 $N$ 个图片。

权重矩阵 $W$ 是一个 $(D,C)$ 尺寸的矩阵, $\omega_{ij}$ 是矩阵的元素

输入 $X$ 是一个 $(N,D)$ 尺寸的矩阵

所以 $S = XW$ 是一个尺寸为 $(N,C)$ 的矩阵，某一行 $s_j$ 代表 $x_i$ 在权重矩阵下的得分

输入 $y$ 是对应输入 $X$ 的正确分类的标签，尺寸为 $(N,)$

损失函数的公式为:

$$
L = \frac{1}{N}\Sigma_i^NL_i + \lambda R(W)
$$

我们的目的是求出 $\frac{\partial{L}}{\partial{W}}$

$$
\frac{\partial{L}}{\partial{W}} = \frac{1}{N}\Sigma_i^N\frac{\partial{L_i}}{\partial{W}} + \lambda\frac{\partial{R(W)}}{\partial{W}}
$$

最终目的求 $L$ 对 $W$ 的梯度，相当于标量对矩阵求导。结果应该和 $W$ 是一样尺寸的 $dW$ 。根据上面的表达式，我们可以先求 $L_i$ 的梯度，再进行求和平均。

那么问题就是如何求出 $\frac{\partial{L_i}}{\partial{W}}$

```math
\frac{\partial{L_i}}{\partial{W}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{11}}} & \frac{\partial{L_i}}{\partial{\omega_{12}}} & \cdots & \frac{\partial{L_i}}{\partial{\omega_{1C}}}\\

\frac{\partial{L_i}}{\partial{\omega_{21}}} & \frac{\partial{L_i}}{\partial{\omega_{22}}} & \cdots & \frac{\partial{L_i}}{\partial{\omega_{2C}}}\\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial{L_i}}{\partial{\omega_{D1}}} & \frac{\partial{L_i}}{\partial{\omega_{D2}}} & \cdots & \frac{\partial{L_i}}{\partial{\omega_{DC}}}\\

\end{pmatrix}
```

根据损失函数的表达式，可以针对 $W$ 的具体一列来求导，比如 $\omega_j$ 或者 $\omega_{y_i}$ 

```math
\frac{\partial{L_i}}{\partial{\omega_j}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{1j}}}\\

\frac{\partial{L_i}}{\partial{\omega_{2j}}}\\

\vdots\\

\frac{\partial{L_i}}{\partial{\omega_{Dj}}}\\

\end{pmatrix}
```

也就是说, $\frac{\partial{L_i}}{\partial{\omega_j}}$ 将会是一个D*1的列向量。

## SVM损失函数求导

对于`SVM`的损失函数`hinge loss`:

$$
L_i = \Sigma_{j \neq y_i}max(0, s_j - s_{y_i} + \Delta) = \Sigma_{j \neq y_i}max(0, x_i\omega_j - x_i\omega_{y_i} + \Delta)
$$

这里 $x_i$ 表示 $X$ 的第 $i$ 行, $\omega_j$ 表示 $W$ 的第 $j$ 列。

那么开始推导 $\frac{\partial{L_i}}{\partial{\omega_j}}$ 很显然，根据 $\Sigma_{j \neq y_i}max(0, x_i\omega_j - x_i\omega_{y_i} + \Delta)$ 这里面只出现了一次 $\omega_j$ 

(注意这里的`j`是具体的一个数，写出 $\omega_0$ $\omega_1$ ... $\omega_j$ ... $\omega_C$ will hlp)

所以

```math
\frac{\partial{L_i}}{\partial{\omega_j}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{1j}}}\\
\\
\frac{\partial{L_i}}{\partial{\omega_{2j}}}\\
\\
\vdots\\
\\
\frac{\partial{L_i}}{\partial{\omega_{Dj}}}\\

\end{pmatrix} =

\left\{
\begin{array}{rcl}
x_i^T & & {x_i\omega_j - x_i\omega_{y_i} + \Delta > 0}\\
\\
0 & & \text{otherwise}\\
\end{array} \right.
```

注意到 $x_i^T$ 是转置的向量，因为标量对向量求导保证 $\omega_j$ 的尺寸一致。

接下来是求 $\frac{\partial{L_i}}{\partial{\omega_{y_i}}}$ 这里 $\omega_{y_i}$ 出现多次, 所以有

```math
\frac{\partial{L_i}}{\partial{\omega_{y_i}}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{1y_i}}}\\
\\
\frac{\partial{L_i}}{\partial{\omega_{2y_i}}}\\
\\
\vdots\\
\\
\frac{\partial{L_i}}{\partial{\omega_{Dy_i}}}\\

\end{pmatrix} =

-\Sigma_{j \neq y_i}

\left\{
\begin{array}{rcl}
x_i^T & & {x_i\omega_j - x_i\omega_{y_i} + \Delta > 0}\\
\\
0 & & \text{otherwise}\\
\end{array} \right.
```

每一个 $\frac{\partial{L_i}}{\partial{\omega_j}}$ 都是 $dW_i$ 的第 $j$ 列 （注意这里 $dW_i$ 是准备后续求和的一部分: $dW = \frac{1}{N}\Sigma_i dW_i + \lambda dR(W)$）

## Softmax损失函数求导

接下来推导`Softmax`的损失函数`cross-entropy loss`:

$$
L_i = -\log{\frac{e^{s_{y_i}}}{\Sigma_je^{s_j}}} = -\log{\frac{e^{x_i\omega_{y_i}}}{\Sigma_je^{x_i\omega_j}}} = -s_{y_i} + \log{\Sigma_je^{s_j}}
$$

(这里的最后一个等号的式子，在编程上可以减少误差积累 PS: log的底数这里为`e`)

分别对 $\omega_j$ 和 $\omega_{y_i}$ 求导

$$
\frac{\partial{L_i}}{\partial{\omega_j}} = - \frac{1}{\frac{e^{x_i\omega_{y_i}}}{\Sigma_je^{x_i\omega_j}}} \cdot (- \frac{x_i^T \cdot e^{x_i\omega_{y_i}} \cdot e^{x_i\omega_j}}{(\Sigma_je^{x_i\omega_j})^2}) = x_i^T \cdot \frac{e^{x_i\omega_j}}{\Sigma_je^{x_i\omega_j}}
$$

$$
\frac{\partial{L_i}}{\partial{\omega_{y_i}}} = x_i^T \cdot (\frac{e^{x_i\omega_{y_i}}}{\Sigma_je^{x_i\omega_j}} - 1)
$$

顺带一提，`assignment 2`中有一个`softmax_loss()`函数需要计算`loss`对`scores`的求导。

这里根据上面的结论以及 $s_j = x_i\omega_j$ 这个式子可以得到:

$$
\frac{\partial{L_i}}{\partial{s_j}} = \frac{\partial{L_i}}{\partial{\omega_j}} \cdot \frac{\partial{\omega_j}}{\partial{s_j}} = \frac{e^{x_i\omega_j}}{\Sigma_je^{x_i\omega_j}}
$$

如果 $j = y_i$ 的话，那就是:

$$
\frac{\partial{L_i}}{\partial{s_{y_i}}} = \frac{e^{x_i\omega_{y_i}}}{\Sigma_je^{x_i\omega_j}} - 1
$$

所以在`softmax_loss()`这里面我是这样写的:

```python
def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    loss = 0.0
    N = x.shape[0]

    # shift消除累计误差
    # 设P为概率Probability = softmax(x)
    P = np.exp(x - x.max(axis = 1)).reshape(N, -1)
    P /= P.sum(axis = 1)
    loss += - np.log(P[range(N), y]).sum() / N
    # loss对score求导 这里scores是x
    P[range(N), y] -= 1
    dx = P / N

    return loss, dx
```

## Momentum Update 理解

一般的SGD (Vanilla SGD)就只是最简单的将位置减去梯度而已:

```python
x = x - learning_rate * dw
```

而**Momentum**方法则是:

```python
v = mu * v - learning_rate * dx
x = x + v
```

可以理解为:

> 将粒子放入目标函数(也就是损失函数)表示的“山坡”上，初速度为v，在山坡上，梯度的方向是山坡增长最快的方向，也就是粒子在此处将受到与梯度相反的作用力，驱使粒子向下运动，同时因为自身的惯性(动量系数mu)，更加平滑地到达低处。

`x`就表示粒子所在的位置，`v`表示粒子的速度

在实际中，一般用`w`权重矩阵来表示粒子在`loss function`的`位置`

## RMSprop 理解

**RMSprop**也就是**Root Mean Square Propagation**

是一种对每个参数的学习率进行自适应调整的优化方法，它通过对每个参数的**梯度平方进行加权平均**来调整学习率。其主要目标是**防止梯度爆炸或梯度消失**现象，并加速训练过程。

同样地，有:

```python
v = decay_rate * v + (1 - decay_rate) * dx ** 2
x += -learning_rate * dx / (np.sqrt(v) + eps)
```

可以理解为:

> 同样将粒子放入目标函数表示的“山坡”上，由于衰退率(decay rate)的影响，粒子将保留原有一部分速度继续运行，另一部分由梯度的平方来提供(平方放大了梯度的影响)，接下来因为速度的影响进而更新位置，步长将会因为sqrt(v)缩放，因为如果v很大，说明这个位置的梯度很大，那么位置更新就会变小；如果v很小，说明这个位置的梯度很小，那么位置更新就会变大，从而更加平滑地到达低处。


ps: `decay_rate`一般取值`[0.9, 0.99, 0.999]`

## Adam 理解

**Adam**也就是**Adaptive Moment Estimate**

Adam结合了Momentum和RMSprop的优点，既有Momentum的方向信息，又有RMSprop的自适应步长。通过引入偏置修正，Adam在训练的初期就能有效地进行更新，并且通过自适应调整学习率，使得训练过程更加稳定高效。

一般写法有:

```python
m = beta1 * m + (1 - beta1) * dx
v = beta2 * v + (1 - beta2) * (dx ** 2)

mt = m / (1 - beta1 ** t)
vt = v / (1 - beta2 ** t)

x += -learning_rate * mt / (np.sqrt(vt) + eps)
```

> 将粒子放入目标函数表示的“山坡”上，粒子不仅受到梯度的影响，还受到其前进方向的惯性影响。首先，m保存了过去梯度的加权平均（即方向信息），v保存了梯度平方的加权平均（即尺度信息）。接下来，在更新时，通过对m和v进行修正（mt和vt），保证了在训练初期的偏置问题，从而使得学习过程更加稳定。

并且因为有`bias correction`机制，并不会像**Adagrad**那样，训练后期学习步长逐渐减小甚至停止学习。

因为这里`v`就算累加了梯度的平方，后面也会修正(`t`是当前的时间步/迭代次数，用于修正`m`和`v`的偏差)

ps: 推荐取值为

```math

\left\{
\begin{array}{l}
\beta_1 = 0.9
\\
\beta_2 = 0.999
\\
eps = 10^{-8}\\
\end{array} \right.
```