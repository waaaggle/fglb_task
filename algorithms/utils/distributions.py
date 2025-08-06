import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical用于离散动作空间的分布
class FixedCategorical(torch.distributions.Categorical):
    #调用父类的 sample() 方法，返回一个采样动作，返回一个形状为 [batch_size, 1] 的动作张量
    def sample(self):
        return super().sample().unsqueeze(-1)

    #计算给定动作的对数概率。，形状也是 [batch_size, 1]
    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    #返回概率最高的动作（即最大概率的类别）。
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal用于连续动作空间的正态分布
class FixedNormal(torch.distributions.Normal):
    #计算动作的对数概率，对连续动作的每个维度计算对数概率后求和（求和是因为每个维度独立）
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    #计算分布的熵，用于衡量分布的不确定性，总熵（对每个维度熵求和）
    def entropy(self):
        return super().entropy().sum(-1)
    #返回分布的众数（对于正态分布，即均值），返回均值 mean，作为正态分布的“最可能动作”
    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    #计算动作的对数概率，对 log prob 做 .view() 操作确保维度对齐后再求和
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    #计算分布的熵
    def entropy(self):
        return super().entropy().sum(-1)

    #返回概率大于 0.5 的动作（即二值化的结果），返回概率大于 0.5 的动作，作为“确定性”动作输出（0 或 1）
    def mode(self):
        return torch.gt(self.probs, 0.5).float()

#用于离散动作空间，输出 FixedCategorical 分布
# num_inputs: 输入特征的维度（如状态嵌入的维度）。
# num_outputs: 动作空间的维度（离散动作的数量）。
# use_orthogonal: 是否使用正交初始化（默认 True）。
# gain: 初始化增益，控制权重初始化的幅度。
class Categorical(nn.Module):
    #选择是否使用正交初始化，并将偏置初始化为 0。
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)  #通过 Linear 层获得 logits；
        #可选：对 available_actions == 0 的 logits 设为 -1e10，使得这些动作概率趋近于 0；
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)  #返回一个 FixedCategorical(logits=logits) 分布对象。

#用于连续动作空间，输出 FixedNormal 分布（对角高斯分布）。
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))  #用于计算每个动作维度的均值；
        self.logstd = AddBias(torch.zeros(num_outputs))  #用于控制每个动作维度的标准差，是一个可学习偏置（用 AddBias 包装）。

    def forward(self, x):
        action_mean = self.fc_mean(x)    #输入 x，得到 action_mean；

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        #用一个 全 0 输入的 dummy tensor，输入到 logstd 层以生成可训练的 log std；
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())  #输出 FixedNormal(mean, std)，用于采样动作。

#用于二值动作空间，输出 FixedBernoulli 分布。
#与 Categorical 类似，也是一个 Linear -> logits -> FixedBernoulli(logits) 的转换过程
class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

#用于为分布的参数添加可学习的偏置。
#输入是 [batch, action_dim]，则 bias 是 [1, action_dim]
#输入是图像（如 [B, C, H, W]），就使用 [1, C, 1, 1] 进行广播
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    #在 forward() 中，它根据输入维度，将偏置广播成适配形状，然后加到输入上。
    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
