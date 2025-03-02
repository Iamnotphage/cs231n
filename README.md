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

[Q2: Batch Normalization](./assignments/assignment2/BatchNormalization.ipynb)

# Notes

这里放一些笔记。主要是矩阵代数相关的内容。

个人整理的[笔记](https://iamnotphage.github.io/blog/2025/%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E7%AC%A6%E5%8F%B7%E5%92%8CNumpy%E8%A7%A3%E9%87%8A/)，整理了关于**各类向量和矩阵乘法**以及对应`numpy`的API

下面会涉及到大量的**标量对向量求导**, **向量对向量求导**, **矩阵对矩阵求导** 等内容。

推荐文章: [矩阵求导术(上)](https://zhuanlan.zhihu.com/p/24709748),[矩阵求导术(下)](https://zhuanlan.zhihu.com/p/24863977), [cs231n-linear-backprop](https://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf)

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

(注意这里的`j`是具体的一个数，写出 $\omega_0$ $\omega_1$ ... $\omega_j$ ... $\omega_C$ will help)

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

## Batch Norm

关于Normalization很简单，就是数据减去其均值再除其标准差。

**Batch Normalization**的理解可以参考[这篇文章](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739/)，此外这是[参考论文](https://arxiv.org/pdf/1502.03167)。其实看完文章基本就掌握了。

简单来说就是在两个隐藏层中间夹一层`Batch Norm`层

在训练阶段，`Batch Norm`层接受来自`Activation`层的输出，首先计算这个`Mini-Batch`的均值和标准差，然后将整个数据**normalize**，然后再进行**Scale & Shift**。同时更新`Moving Average`(后续推理阶段使用)

> At training time, such a layer uses a minibatch of data to estimate the mean and standard deviation of each feature. These estimated means and standard deviations are then used to center and normalize the features of the minibatch. A running average of these means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.

![batchnorm_graph](./assignments/assignment2/cs231n/notebook_images/batchnorm_graph.png)

最复杂的是推导Batch Norm反向传播的一系列矩阵

先规定一些符号:

```math
\begin{aligned}

X = \begin{pmatrix}

X_1\\

X_2
\\
\vdots\\

X_N

\end{pmatrix}
,
X_i = \begin{bmatrix} x_{i1} & x_{i2} & \cdots & x_{iD} \end{bmatrix}

\\

\gamma = \begin{bmatrix} \gamma_1 & \gamma_2 & \cdots & \gamma_D \end{bmatrix}

,

\beta = \begin{bmatrix} \beta_1 & \beta_2 & \cdots & \beta_D \end{bmatrix}

\\

Y = \begin{pmatrix}

Y_1\\

Y_2
\\
\vdots\\

Y_N

\end{pmatrix}
,
Y_i = \begin{bmatrix} y_{i1} & y_{i2} & \cdots & y_{iD} \end{bmatrix}

\end{aligned}
```

$X$ 和 $\hat{X}$ 以及 $Y$ 都是 (N,D) 的矩阵。 $\gamma$ 和 $\beta$ 这里设置为 $1 \times D$ 的行向量

根据计算图，有:

```math
\begin{aligned}

\mu = \frac{1}{N}\sum_{k=1}^NX_k
\\
v = \frac{1}{N}\sum_{k=1}^N(X_k - \mu)^2
\\
\sigma = \sqrt{v + \epsilon}
\\
\hat{X_i} = \frac{X_i - \mu}{\sigma}
\\
Y_i = \gamma \bigodot \hat{X_i} + \beta

\end{aligned}
```

这里需要注意两点:

1. 计算图没给出 $\hat{X_i}$ 这里计算用到了
2. $\bigodot$ 或者 $\circ$ 指的是`element-wise multiply`对应`numpy`的`*`运算符或`np.multiply()` 具体查看`Hadamard product`相关词条

---

接下来求 $\frac{\partial{L}}{\partial{X}}$ 、$\frac{\partial{L}}{\partial{\gamma}}$ 和 $\frac{\partial{L}}{\partial{\beta}}$

先来个简单的 $\frac{\partial{L}}{\partial{\beta}}$ 练手。

首先是

```math
\frac{\partial{L}}{\partial{\beta}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{\beta}}
```

其中 $\frac{\partial{L}}{\partial{Y_i}}$ 是上游传过来的`dout`的一部分。没错，这也是那篇论文的公式，但是，为什么？

一般到这里有两个疑点:

1. 为什么要求和？What the hell？
2. 为什么不是 $\frac{\partial{L}}{\partial{\beta}} = \frac{\partial{L}}{\partial{Y}} \frac{\partial{Y}}{\partial{\beta}}$

我在这里卡了非常非常久，后面发现其实很简单！

因为本质上 $L = f(Y_1, Y_2, Y_3, \dots, Y_N)$ 是 $Y_i$ 的函数。也就是多个变量的函数，只是在计算的时候利用了`numpy`的广播机制才写出来`out = gamma * x_hat + beta`的代码，实际上是每一个 $Y_i = \gamma \bigodot \hat{X_i} + \beta$，然后多个 $Y_i$ 组合成一个矩阵 $Y$ 再交给下游的 $f$ 处理。但是下游 $f$ 实际上也只是处理 $Y_i$ 而已。

所以根据上面的关于 $L = f(Y_i)$ 的理解:

```math
\frac{\partial{L}}{\partial{\beta}} = \frac{\partial{L}}{\partial{Y_1}}\frac{\partial{Y_1}}{\partial{\beta}} + \frac{\partial{L}}{\partial{Y_2}}\frac{\partial{Y_2}}{\partial{\beta}} + \dots + \frac{\partial{L}}{\partial{Y_N}}\frac{\partial{Y_N}}{\partial{\beta}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{\beta}}
```

所以就转而求 $\frac{\partial{Y_i}}{\partial{\beta}}$ 了

这里介绍两个方法求 $\frac{\partial{Y_i}}{\partial{\beta}}$ (向量对向量求导)

**第一个方法**: 逐个元素求导，也就是穷举出所有的组合，然后组成一个矩阵(这里不用太纠结分子布局还是分母布局)

```math
\begin{aligned}

Y_i = \gamma \bigodot \hat{X_i} + \beta = 

\begin{pmatrix} y_{i1} \\ y_{i2} \\ y_{i3} \\ \vdots \\ y_{iD} \end{pmatrix}^T = 

\begin{pmatrix} \gamma_1\hat{x_{i1}}+\beta_1 \\ \gamma_2\hat{x_{i2}}+\beta_2 \\ \gamma_3\hat{x_{i3}}+\beta_3 \\ \vdots \\ \gamma_D\hat{x_{iD}}+\beta_D \end{pmatrix}^T

\\

\frac{\partial{Y_i}}{\partial{\beta}} = 

\begin{pmatrix}

\frac{\partial{y_{i1}}}{\partial{\beta_1}} & \frac{\partial{y_{i2}}}{\partial{\beta_1}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\beta_1}}

\\

\frac{\partial{y_{i1}}}{\partial{\beta_2}} & \frac{\partial{y_{i2}}}{\partial{\beta_2}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\beta_2}}

\\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial{y_{i1}}}{\partial{\beta_D}} & \frac{\partial{y_{i2}}}{\partial{\beta_D}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\beta_D}}

\end{pmatrix}

= 

\begin{pmatrix}

1 & 0 & \cdots & 0

\\

0 & 1 & \cdots & 0

\\

\vdots & \vdots & \ddots & \vdots \\

0 & 0 & \cdots & 1

\end{pmatrix} = I_D

\end{aligned}
```

实际上: $\frac{\partial{y_{ij}}}{\partial{\beta_k}} = 1$ 当且仅当 $j = k$ 

**第二个方法**: 根据[矩阵求导术](https://zhuanlan.zhihu.com/p/24863977)，先求全微分，再根据矩阵相关的变换规则，求出对应的导数。不过这里显得很简单。

```math
\mathrm{d} Y_i = \mathrm{d} (\gamma \circ \hat{X_i} + \beta) = \mathrm{d} \beta
\\
\text{vec}(\mathrm{d} Y_i) = \text{vec}(\mathrm{d} \beta) = I_D \cdot \text{vec}(\mathrm{d} \beta)
```

$\text{vec}(X)$ 表示对矩阵 $X$ 按列优先向量化, 所以最后等式左右两边是 $D \times 1$ 尺寸，添加一个单位矩阵的技巧不影响。

```math
\text{vec}(\mathrm{d} Y_i) = \frac{\partial{Y_i}}{\partial{\beta}}^T \text{vec}(\mathrm{d} \beta )
```

得到

```math
\frac{\partial{Y_i}}{\partial{\beta}} = I_D
```

结果一样。

所以最后答案是一个对角线全 $1$ 尺寸为 $D \times D$ 的单位矩阵

```math
\frac{\partial{L}}{\partial{\beta}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{\beta}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} I_D = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}}
```

---

继续求 $\frac{\partial{L}}{\partial{\gamma}}$

```math
\frac{\partial{L}}{\partial{\gamma}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}}\frac{\partial{Y_i}}{\partial{\gamma}}
```

因为链式法则,进一步求 $\frac{\partial{Y_i}}{\partial{\gamma}}$

**第一个方法**: 逐个元素求导，也就是穷举出所有的组合，然后组成一个矩阵

```math
\frac{\partial{Y_i}}{\partial{\gamma}} = 

\begin{pmatrix}

\frac{\partial{y_{i1}}}{\partial{\gamma_1}} & \frac{\partial{y_{i2}}}{\partial{\gamma_1}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\gamma_1}}

\\

\frac{\partial{y_{i1}}}{\partial{\gamma_2}} & \frac{\partial{y_{i2}}}{\partial{\gamma_2}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\gamma_2}}

\\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial{y_{i1}}}{\partial{\gamma_D}} & \frac{\partial{y_{i2}}}{\partial{\gamma_D}} & \cdots & \frac{\partial{y_{iD}}}{\partial{\gamma_D}}

\end{pmatrix}

= 

\begin{pmatrix}

x_{i1} & 0 & \cdots & 0

\\

0 & x_{i2} & \cdots & 0

\\

\vdots & \vdots & \ddots & \vdots \\

0 & 0 & \cdots & x_{iD}

\end{pmatrix} = \text{diag}(\hat{X_i})
```

**第二个方法:**

先全微分，再利用矩阵相关法则，转成 $\text{vec}(\mathrm{d}Y_i) = \frac{\partial{Y_i}}{\partial{\gamma}}^T \text{vec}(\mathrm{d} \gamma)$ 根据对应关系求出 $\frac{\partial{Y_i}}{\partial{\gamma}}$

```math
\mathrm{d} Y_i = \mathrm{d}(\gamma \circ \hat{X_i})
```

进一步推导(注意Hadamard乘积满足**交换律**):

```math
\begin{aligned}

\text{vec}(\mathrm{d} Y_i) &= \text{vec}(\mathrm{d} (\gamma \circ \hat{X_i}))\\

&= \text{vec}(\mathrm{d}\gamma \circ \hat{X_i} + \gamma \circ \mathrm{d}\hat{X_i})\\

&= \text{vec}(\mathrm{d}\gamma \circ \hat{X_i})\\

&= \text{vec}(\hat{X_i} \circ \mathrm{d}\gamma)\\

&= \text{diag}(\hat{X_i}) \text{vec}(\mathrm{d}\gamma)

\end{aligned}
```

其中， $\text{diag}(\hat{X_i})$ 是将 $X_i$ 的元素，按照列优先组成的对角阵( $D \times D$ )

所以:

```math
\begin{aligned}

\frac{\partial{Y_i}}{\partial{\gamma}} = \text{diag}(\hat{X_i}) = 

\begin{pmatrix}

x_{i1} & 0 & \cdots & 0

\\

0 & x_{i2} & \cdots & 0

\\

\vdots & \vdots & \ddots & \vdots \\

0 & 0 & \cdots & x_{iD}

\end{pmatrix}

\\

\frac{\partial{L}}{\partial{\gamma}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}}\frac{\partial{Y_i}}{\partial{\gamma}} = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} \text{diag}(\hat{X_i}) = \sum_{i=1}^{N}\frac{\partial{L}}{\partial{Y_i}} \circ \hat{X_i}

\end{aligned}
```

最后乘对角阵就相当于和 $\hat{X_i}$ 逐个元素相乘(Hadamard乘积)

---

求 $\frac{\partial{L}}{\partial{X}}$ 时，同样的道理。因为每次传播只是对多个样本进行运算，也就是每次计算一个 $X_i$ 。只不过实际写代码时可以直接将 $X_i$ 组合成矩阵 $X$。所以我们得推导 $\frac{\partial{L}}{\partial{X_i}}$。

因为 $L = f(Y_1, Y_2, Y_3, \dots, Y_N)$ 是 $Y_i$ 的函数。而 $Y_i$ 跟 $\gamma$, $\hat{X_i}$, $\beta$ 有关系

求 $\frac{\partial{L}}{\partial{\gamma}}$ 和 $\frac{\partial{L}}{\partial{\beta}}$ 时，因为每个 $Y_i$ 的 $\gamma$ 和 $\beta$ 都是一样的，所以要每个 $Y_i$ 都求导最后求和。

而求 $\frac{\partial{L}}{\partial{X_i}}$ 时，比如 $\frac{\partial{L}}{\partial{X_3}}$ 只和 $Y_3$ 有关系，所以可以直接写

```math
\frac{\partial{L}}{\partial{X_i}} = \frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{X_i}}
```

当然也可以理解为: 因为其余项和 $X_i$ 没关系，所以导数为 $0$ 最后求和仍然是单项:

```math
\frac{\partial{L}}{\partial{X_i}} = \frac{\partial{L}}{\partial{Y_1}} \frac{\partial{Y_1}}{\partial{X_i}} + \frac{\partial{L}}{\partial{Y_2}} \frac{\partial{Y_2}}{\partial{X_i}} + \dots + \frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{X_i}} + \dots + \frac{\partial{L}}{\partial{Y_N}} \frac{\partial{Y_N}}{\partial{X_i}} = \frac{\partial{L}}{\partial{Y_i}} \frac{\partial{Y_i}}{\partial{X_i}}
```

于是，我们的问题转而求 $\frac{\partial{L}}{\partial{X_i}}$

 