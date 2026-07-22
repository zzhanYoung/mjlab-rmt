# ROAM: Reduced-Order Observer-Augmented Motion Tracking

> 本报告记录了从初始设计到实验验证的全部设计决策、踩坑经验和关键洞见，供后期论文写作参考。

---

## 一、研究概览

本工作聚焦于Humanoid Motion Tracking中的Sim2Real Gap/Distribution Shift问题，用Disturbance Observer估计残差信号并经过可学习滤波器（Optional）处理后，放入Observation，加强Policy的Robustness. 针对pelvis无法获得速度信号的问题，打算将Whole-body dynamics reduce-order，假设基座的速度被固定，但是姿态自由（被一个虚拟的球头铰链限制住自由度）。

### Full-Order Whole-body Disturbance Observer

Humanoid浮动基座动力学同样采用Euler-Lagrange形式构建（包括浮动基）:

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = S^\top \tau + J(q)^T f_{con} + \tau_r
$$

其中：

- $f_{con}$为足端接触力
- $q$对应所有广义位置（浮动基座位姿以及子关节的角度）
- $\tau$为关节驱动力
- $\tau_r$为未建模残差力矩、外部载荷带来的力矩

现在我们设计Disturbance Observer：

$$
d = J(q)^T f_{con} + \tau_r
$$

Momentum一般形式（一个嵌入动力学的LPF）：

$$
\dot{\hat{d}} = L(d - \hat{d})
$$

$$
\dot{\hat{d}} = L(M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) - S^\top \tau - \hat{d})
$$

定义辅助变量：

$$
\dot{\xi} = \dot{\hat{d}} - L\frac{d}{dt}[M(q)\dot{q}] = \dot{\hat{d}} - L[M(q)\ddot{q} + C(q,\dot{q})\dot{q} + C(q,\dot{q})^\top\dot{q}]
$$

积分得到：

$$
\xi = \hat{d} - LM(q)\dot{q}
$$

带回：

$$
\begin{aligned}
\dot{\xi} &= \dot{\hat{d}} - L[M(q)\ddot{q} + C(q,\dot{q})\dot{q} + C(q,\dot{q})^\top\dot{q}]\\
&= L(M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) - S^\top \tau - \hat{d}) \\&\quad-L[M(q)\ddot{q} + C(q,\dot{q})\dot{q} + C(q,\dot{q})^\top\dot{q}]\\
&= L(-C(q,\dot{q})^\top\dot{q} + G(q) - S^\top \tau - \hat{d})
\end{aligned}
$$

其中，$-C(q,\dot{q})^\top\dot{q} + G(q)$采用$q_{bias} = C(q,\dot{q})\dot{q} + G(q)$近似，这样的近似等价于$\dot{M}(q) \approx 0$的假设，即可得到Estimated Disturbance表达式。

### Reduced-Order / Rotation-Only Whole-body Dynamics

我们假设浮动基座没有速度，只有姿态（被一个虚拟的球头铰链固定），减少三个自由度，每次调用`mjlab`环境进行计算$M, q_{bias}$时，我们直接覆盖基座广义速度，将translational分量设置为全0：

$$
v_f = [0, 0, 0, w_x, w_y, w_z]^\top
$$

在这个基础上，mjlab重新计算$M, q_{bias}$，并切片去掉基座的translational自由度，即可获得reduced-order dynamics。这是我们估计出来的残差本质上还有这样假设带来的误差，我们假设这样的残差信号依然是有利于基座控制的。估计出来的残差力矩出现在各个关节，以及浮动基座的旋转自由度上（忽略水平加速度）。