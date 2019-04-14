# 机器学习概述

## 分析为什么平方损失函数不适用于分类问题

首先证明平方损失函数的梯度为：

$$ dR = tr(d(y-X^TW)^T(y-X^TW) + (y-X^TW)^Td(y-X^TW))  $$

$$ dR = tr(-(X^TdW)^T(y-X^TW) - (y-X^TW)^TX^TdW)) $$

$$ dR = tr(-2(y - X^TW)^TX^TdW) $$

因此：

$$ \frac{\partial{R}}{\partial{W}} = 2X(X^TW-y) $$

$$ XX^TW=Xy  $$

$$ W = (XX^T)^{-1}Xy $$

由于分类问题中向量y只有1维等于1，因此若数据集中大部分都是其中1类，则会造成模式崩坏问题。

## 在线性回归中，如果我们给每个样本 $ (x^{(n)}, y^{(n)}) $ 赋予一个权重$r^{(n)}$，经验风险函数为$$ R(w) = \frac{1}{2} \sum_{n=1}^N r^{(n)}(y^{(n)} - w^Tx^{(n)})^2 $$，计算其最优参数$ w^* $，并分析权重$r^{(n)}$ 的作用。

https://blog.csdn.net/zhengjihao/article/details/70318660

## 在线性回归中，验证岭回归的解为结构风险最小化准则下的最小二乘法估计，见公式(2.45)

$$ R = \frac{1}{2} \parallel y-X^Tw \parallel ^2 + \frac{1}{2} \lambda \parallel w \parallel ^2 $$

$$ dR = 2tr((w^TX - y)X^Tdw) + 2tr(\lambda (dw^Tw + w^Tdw)) $$

$$ dR = 2tr((w^TX - y)X^Tdw) + 2\lambda  tr(w^Tdw) $$

$$ dR = 2tr([(w^TX - y)X^T + \lambda w^T]dw) $$

因此：

$$ \frac{\partial R}{\partial w} = 2X(X^Tw - y) + 2\lambda w $$

令 $ \frac{\partial R}{\partial w} = 0$，则有

$$ XX^Tw -Xy + \lambda w = 0 $$

$$ (XX^T + \lambda I)w = Xy $$

$$ w = (XX^T + \lambda I)^{-1}Xy $$