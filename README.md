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

# Notes

简单推导一下`SVM`和`Softmax`的损失函数对权重矩阵`W`的梯度。

首先明确一些符号。

一张图像一共有 $D$ 个像素，可以被分类为 $C$ 个类别，一共有 $N$ 个图片。

权重矩阵 $W$ 是一个 $(D,C)$ 尺寸的矩阵

输入 $X$ 是一个 $(N,D)$ 尺寸的矩阵

输入 $y$ 是对应输入 $X$ 的正确分类的标签，尺寸为 $(N,)$

损失函数的公式为:

$$
L = \frac{1}{N}\Sigma_i^NL_i + \lambda R(W)
$$

对于`SVM`的损失函数`hinge loss`:

$$
L_i = \Sigma_{j \neq y_i}max(0, s_j - s_{y_i} + \Delta) = \Sigma_{j \neq y_i}max(0, x_i\omega_j - x_i\omega_{y_i} + \Delta)
$$

这里 $x_i$ 表示 $X$ 的第 $i$ 行, $\omega_j$ 表示 $W$ 的第 $j$ 列。

最终目的求 $L$ 对 $W$ 的梯度，相当于标量对矩阵求导。结果应该和 $W$ 是一样尺寸的 $dW$ 。根据 $L$ 的表达式，我们可以先求 $L_i$ 的梯度，再进行求和平均。

根据损失函数的表达式，可以针对 $W$ 的具体一列来求导，比如 $\omega_j$ 或者 $\omega_{y_i}$ 

```math
\frac{\partial{L_i}}{\partial{\omega_j}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{0j}}}\\

\frac{\partial{L_i}}{\partial{\omega_{1j}}}\\

...\\

\frac{\partial{L_i}}{\partial{\omega_{Dj}}}\\

\end{pmatrix}
```

也就是说, $\frac{\partial{L_i}}{\partial{\omega_j}}$ 将会是一个D*1的列向量。

那么开始推导 $\frac{\partial{L_i}}{\partial{\omega_j}}$ 很显然，根据 $\Sigma_{j \neq y_i}max(0, x_i\omega_j - x_i\omega_{y_i} + \Delta)$ 这里面只出现了一次 $\omega_j$ 

(注意这里的`j`是具体的一个数，写出 $\omega_0$ $\omega_1$ ... $\omega_j$ ... $\omega_C$ will hlp)

所以

```math
\frac{\partial{L_i}}{\partial{\omega_j}} = \begin{pmatrix}

\frac{\partial{L_i}}{\partial{\omega_{0j}}}\\
\\
\frac{\partial{L_i}}{\partial{\omega_{1j}}}\\
\\
...\\
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

\frac{\partial{L_i}}{\partial{\omega_{0y_i}}}\\
\\
\frac{\partial{L_i}}{\partial{\omega_{1y_i}}}\\
\\
...\\
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

---

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
