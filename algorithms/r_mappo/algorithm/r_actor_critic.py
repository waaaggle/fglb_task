import torch
import torch.nn as nn
from fglb_task.algorithms.utils.util import init, check
from fglb_task.algorithms.utils.cnn import CNNBase
from fglb_task.algorithms.utils.mlp import MLPBase
from fglb_task.algorithms.utils.rnn import RNNLayer
from fglb_task.algorithms.utils.act import ACTLayer
from fglb_task.algorithms.utils.popart import PopArt
from fglb_task.utils.util import get_shape_from_obs_space

#策略网络，输入观测，输出动作分布，用于决策。
class R_Actor(nn.Module):
    """
    Actor fglb_dut class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    # args：各种超参数和开关，决定网络结构、初始化方式、是否RNN等。
    # obs_space：输入观测空间，可以决定输入shape和基网络类型（CNN / MLP）。
    # action_space：动作空间，用于决定输出分布形状和类型。
    # device：运行设备(cpu / gpu)。
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)  #字典，指定张量默认的数据类型和设备，便于后续.to(**self.tpdv)调用。

        obs_shape = get_shape_from_obs_space(obs_space)
        #特征提取基网络。若观测为图像(3维)，用CNNBase，否则用MLPBase
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        #可选，若配置为RNN策略，则加一层RNN（如GRU/LSTM/RNN），GRU门控神经网络，RNN的改进变体
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        #动作输出层（ACTLayer），根据动作空间生成动作分布并输出采样/概率。
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    # obs, rnn_states, masks, available_actions：输入观测、RNN隐藏状态、mask（用于RNN重置）、可用动作（部分动作失效时用）。
    # deterministic：是否用分布均值（测试 / 评估），还是采样（训练）。
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into fglb_dut.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN fglb_dut, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        #如果用RNN，则处理RNN，得到新的特征和隐藏状态
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        #返回动作、动作对数概率、更新后的RNN状态。
        return actions, action_log_probs, rnn_states

    #用于评估给定动作的对数概率和分布熵（用于PPO等算法的损失计算）。
    #如果配置use_policy_active_masks，则只对存活/激活的智能体评估概率。
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into fglb_dut.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN fglb_dut, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        #支持HATRPO算法的特殊evaluate接口。
        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy

#价值网络，输入（中心化）观测，输出价值估计（V值）。
class R_Critic(nn.Module):
    """
    Critic fglb_dut class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    # args：超参数
    # cent_obs_space：中心化观测空间（全局信息，用于MAPPO / VDN等集中训练）
    # device：设备
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        #特征提取基网络（CNN/MLP）
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        #可选RNN层
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        #网络参数初始化方法（Xavier或正交初始化）
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        #输出层，将特征转换为value。支持PopArt归一化（自适应value输出归一化，提升训练稳定性）。
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into fglb_dut.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN fglb_dut, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        #如果配置用RNN，则进入RNN层
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        #输出value（V(s))和新的RNN状态
        return values, rnn_states
