# 网络优化和正则化

1. 证明在动量法的更新公式(7.15)中，∆θt实际上是相当于对$ -\frac{\alpha}{1-p} g_t $进行指数衰减的移动平均。

    $$ \Delta\theta_t = \rho \theta_{t-1} - \alpha g_t \\ = \rho \theta_{t-1} - (1-\rho) \frac{\alpha}{1-\rho} g_t $$

2. 在Adam算法中，说明指数加权平均的偏差修正公式(??)和(??)的合理性。

    $$ M_t = \beta_1M_{t-1} + (1 - \beta_1)g_t \\ = (1-\beta_1)\sum_{\dag} ^t \beta_1^{t-\dag}g_t $$ 

    $$ \hat{M_t} = \frac{M_t}{1-\beta_1^t}= \frac{(1-\beta_1)\sum_\dag^t\beta_1^{t-\dag}g_t}{1-\beta_1^t} $$

    于是$ M_t $ 可被认为是梯度的均值（一阶矩），在早期会很小，与均值的偏差较大,而$ \hat{M_t} $早期被放大了。
    且：

    $$ \lim_{t \to \infty}\hat{M_t} = M_t  $$

    同样的情况对于 $ G_t $也是如此

3. 分析为什么**批量归一化**不能直接应用于循环神经网络。
    因为循环神经网络的净输入的分布是动态变化的
4. 证明在标准的随机梯度下降中，权重衰减正则化和ℓ2 正则化的效果相同。并分析这一结论在动量法和Adam算法中是否依然成立。
   - 当使用ℓ2时，每次更新梯度为：
    $$ \theta_t = \theta_{t-1} - \alpha(g_t + \lambda \theta_{t-1}) \\ = (1 -\alpha\lambda)\theta_{t-1} - \alpha g_t  $$
    将$ \alpha \lambda $ 看作参数$ w $，即它们的效果相同。
    - 在动量法中
     ℓ2正则化的梯度更新公示变为：
     $$ \theta_t = \rho \theta_{t-1} - \alpha(g_t + \lambda \theta_{t-1}) \\ =  (\rho -\alpha\lambda)\theta_{t-1} - \alpha g_t   $$
     而权重衰减的梯度更新公式为：
     $$ \theta_t = \rho(1 - w)\theta_{t-1} - \alpha g_t   $$
     仍旧成立
    - 在Adam算法中
      由于$ \alpha $ 变为了与t相关的量，不再是固定值，因此不能等效。
5. 试分析为什么不能在循环神经网络中的循环连接上直接应用丢弃法？
   这样会损害循环网络在时间维度上记忆能力。
6. 若使用标签平滑正则化方法，给出其交叉熵损失函数。
   $$ loss = -\sum_{i=1}^m (\frac{\epsilon}{K-1} + (1-\epsilon) \cdot y) log\hat{y}  $$
