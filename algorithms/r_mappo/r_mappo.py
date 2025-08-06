"""
MAPPO（Multi-Agent Proximal Policy Optimization）算法的训练器（Trainer）核心实现，
负责管理策略网络（policy）的训练过程，包括损失计算、参数更新、优势归一化、minibatch采样等。

初始化：创建 R_MAPPO 实例，配置策略和超参数。
数据收集：使用 prep_rollout 收集数据，存入 SharedReplayBuffer。
训练：
调用 prep_training。
执行 train，处理缓冲区数据，更新 actor 和 critic。

监控：通过 train_info 查看损失、梯度范数等指标。

"""
import numpy as np
import torch
import torch.nn as nn
from fglb_task.utils.util import get_gard_norm, huber_loss, mse_loss
from fglb_task.utils.valuenorm import ValueNorm
from fglb_task.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    #初始化 R_MAPPO 训练器，配置超参数、策略和计算设备。
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # self.tpdv：指定张量类型
        self.policy = policy    #一个封装了 actor-critic 网络的对象

        #PPO 超参数加载
        self.clip_param = args.clip_param  #PPO 剪切参数（如 0.2），控制更新幅度。
        self.ppo_epoch = args.ppo_epoch #数据批次迭代次数。
        self.num_mini_batch = args.num_mini_batch #每轮的小批量数。
        self.data_chunk_length = args.data_chunk_length  #用于循环策略的数据块长度。
        self.value_loss_coef = args.value_loss_coef  #价值损失的权重
        self.entropy_coef = args.entropy_coef  #熵正则化的权重，促进探索
        self.max_grad_norm = args.max_grad_norm       #最大梯度范数，防止梯度爆炸
        self.huber_delta = args.huber_delta  #Huber 损失的 delta 参数

        #训练策略标志位
        self._use_recurrent_policy = args.use_recurrent_policy  #是否用循环网络
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart  #PopArt 归一化
        self._use_valuenorm = args.use_valuenorm  #值归一化
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        #值归一化选择器
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    #计算价值函数损失
    # values：当前critic的价值预测。
    # value_preds_batch：数据批次中的旧价值预测（用于剪切）。
    # return_batch：实际回报（例如折扣奖励或GAE回报）。
    # active_masks_batch：二进制掩码，指示智能体在某时间步是否活跃（1表示活跃，0表示死亡 / 不活跃）。
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        #将当前预测与旧预测的差值限制在 [-clip_param, clip_param] 内，遵循 PPO 的保守更新策略：
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        #归一化
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            #计算误差
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        #损失计算
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        #剪切 vs. 非剪切损失
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        #掩码处理
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    #使用数据批次执行 actor 和 critic 网络的单次更新
    # sample：包含训练数据的元组。
    # update_actor：是否更新actor网络。
    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor fglb_dut.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        # 检查sample长度（12或13，可能兼容不同缓冲区格式），解包为：
        # share_obs_batch：共享观测（用于集中式critic）。
        # obs_batch：单个智能体观测。
        # rnn_states_batch, rnn_states_critic_batch：actor和critic的循环网络状态。
        # actions_batch：执行的动作。
        # value_preds_batch：旧价值预测。
        # return_batch：回报。
        # masks_batch：有效时间步掩码。
        # active_masks_batch：活跃智能体掩码。
        # old_action_log_probs_batch：旧策略的动作对数概率。
        # adv_targ：优势估计（如GAE）。
        # available_actions_batch：可用动作（离散动作空间）。
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        #使用 check 函数（来自 onpolicy.utils.util）将关键张量转换为指定设备和类型
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        #调用 self.policy.evaluate_actions 获取当前价值预测 (values)、动作对数概率 (action_log_probs) 和动作分布熵 (dist_entropy)。
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        #计算重要性采样权重
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        #计算 PPO 剪切目标
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        #添加熵正则化，若 update_actor 为 True，执行反向传播并更新 actor
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        #梯度剪切
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        #使用 cal_value_loss 计算价值损失，乘以 value_loss_coef 后反向传播
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        #返回 value_loss、 critic_grad_norm、 policy_loss、 dist_entropy、 actor_grad_norm 和 imp_weights。
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    #使用小批量梯度下降执行多轮训练更新
    # buffer：SharedReplayBuffer实例，包含训练数据。
    # update_actor：是否更新actor网络。
    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor fglb_dut.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        #计算优势（回报与价值预测的差值），若使用 PopArt 或 ValueNorm，先对价值预测去归一化
        #归一化优势为零均值、单位方差，忽略不活跃智能体
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            #根据策略类型选择数据生成器
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            #生成器产生小批量数据
            #迭代 ppo_epoch 次，处理每个小批量
            for sample in data_generator:

                #调用 ppo_update 计算损失并更新网络
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        #将累积指标除以总更新次数（ppo_epoch * num_mini_batch）
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        #返回 train_info，包含平均损失、梯度范数等
        return train_info

    #将 actor 和 critic 网络设置为训练模式（启用 dropout、批归一化等）。
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    #设置为评估模式（禁用训练特定行为）。确保网络在训练或推理（RL 中的 rollout）时处于正确模式。
    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
