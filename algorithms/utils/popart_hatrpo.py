"""
这段代码是另一个版本的 PopArt 实现，主要用于对观测值（或目标值）进行标准化，但不带权重重映射功能（不像前一版那样自动调整线性层参数）。
它主要实现了自适应地估计均值和方差，并提供标准化和去标准化功能，常用于强化学习中的目标值或奖励值的标准化。
"""
import numpy as np

import torch
import torch.nn as nn


class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""
    # input_shape:输入数据的维度，如 (3,) 表示每个输入是一个 3 维向量
    # norm_axes：指定对哪几个维度进行归一化统计。默认是 1，表示按 batch 维度（第 0 维）做均值方差统计。
    # beta：EMA 衰减系数，值越接近 1，更新越平滑、越慢。
    # per_element_update：是否按照元素级别调整 beta 权重（对 batch size 敏感）。
    # epsilon：防止除零的小常数。
    # device：模型运行设备。
    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        #这三个变量都不会反向传播梯度（requires_grad=False），是为了追踪数据统计量而存在的。
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)  # 跟踪均值
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)  # 跟踪平方均值
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)  # 纠正EMA初期偏差

    #将统计量重置为 0，用于模型重新初始化时调用：
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    #计算 去偏的（debised）均值和方差：注意：这一步防止方差为负或为 0，确保数值稳定。
    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)  #转换为张量
        input_vector = input_vector.to(**self.tpdv)

        #如果处于训练模式，就更新统计量
        if train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()

            #计算当前 batch 的均值和平方均值
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            #是否使用“每元素更新策略”（batch size 感知的 beta），这种策略会根据 batch 的大小动态调整更新速度。
            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            #使用 EMA 更新全局统计量
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        #使用去偏统计量进行标准化
        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    #就是对 forward() 的调用，便于使用。
    def normalize(self, input_vector, train=True):
        return self.forward(input_vector, train)

    #把标准化后的数据恢复为原始分布：
    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        #并返回 CPU 上的 numpy 数组（用于日志、可视化或进一步处理）：
        out = out.cpu().numpy()
        
        return out
