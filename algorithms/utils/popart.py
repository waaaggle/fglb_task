"""
PopArt（Population Artihmetic）是一种自适应标准化机制，用于解决强化学习中目标尺度变化剧烈的问题，尤其适用于值函数（如 V 值、Q 值）的学习
PopArt 是一个特殊的线性层，它能跟踪输出的均值和方差，在每次更新这些统计量时自动调整权重和偏置，以保证标准化不会改变网络输出的实际语义。
它在强化学习中特别重要，能让目标尺度变化较大的问题（如不同环境 reward 范围）变得更稳定。
功能：
对输出进行标准化处理；
追踪并更新目标值的统计特征（均值和方差）；
动态调整神经网络的最后一层权重和偏置以保持输出不变形。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(torch.nn.Module):
    # input_shape：输入特征的维度（如状态嵌入的维度）。
    # output_shape：输出值的维度（如值函数的输出，通常为1或多维）。
    # norm_axes：沿哪些轴计算均值和方差（默认1，表示对批次维度标准化）。
    # beta：指数移动平均的衰减因子，用于更新均值和方差（默认0.99999，接近1，表示缓慢更新）。
    # epsilon：防止除零的小值（默认1e-5）。
    # device：张量所在的设备（默认cpu）。
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)  # 标准差（从 mean 和 mean_sq 推导）
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)  # 输出的均值
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)  # 输出的平方均值
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)   # 用于修正EMA初始偏差

        self.reset_parameters()

    #初始化线性层的权重和偏置，以及统计量参数
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  #标准初始化线性层的权重和偏置：
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    #作为值函数网络的最后一层，输出值函数的估计（如状态值或动作值）。
    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    #更新统计量
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        #获取旧统计量（用于之后调整参数）
        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        #计算新的批量均值 & 平方均值（不进行去偏处理）
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        #用指数移动平均更新全局均值 & 方差
        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        #重新计算标准差
        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        #更新线性层权重和偏置（保持“非标准化输出”值不变）
        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        #虽然我们标准化了输出，但模型的整体输出（反标准化后）保持不变！
        self.weight = self.weight * old_stddev / new_stddev
        self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    #去偏均值和方差
    def debiased_mean_var(self):
        #EMA 在初期会有偏差，所以需要去偏（偏差修正）：
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        #然后计算无偏方差：
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    #将输入值标准化为均值 0、方差 1 的形式
    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    #将标准化值恢复成原始尺度
    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()

        return out
