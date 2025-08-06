import numpy as np
import gym
from gym import spaces

class GymFglbEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_agents=4, episode_length=200, share_reward=True, dynamic=True):
        super(GymFglbEnv, self).__init__()
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.share_reward = share_reward
        self.dynamic = dynamic

        self.step_count = 0
        self.device_loads = np.zeros(self.num_agents, dtype=np.float32)  #所有agent的负载
        self.active_mask = np.ones(self.num_agents, dtype=np.float32)  #所有agent的mask

        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(self.num_agents + 1)
            for i in range(self.num_agents)
        })
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
            for i in range(self.num_agents)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.device_loads = np.random.rand(self.num_agents).astype(np.float32)
        self.active_mask = np.ones(self.num_agents, dtype=np.float32)
        return self._get_obs(), {}

    #actions代表agent采取一个action，格式为：agent_id, act
    def step(self, actions: dict):
        self.step_count += 1
        if self.dynamic:
            self._simulate_node_status()
            self._generate_tasks()

        for agent_id, act in actions.items():
            i = int(agent_id.split("_")[1])
            if self.active_mask[i] == 0 or act == 0:
                continue
            target = act - 1
            if target == i or self.active_mask[target] == 0:
                continue
            delta = self.device_loads[i] * 0.2
            self.device_loads[i] -= delta
            self.device_loads[target] += delta

        obs = self._get_obs()
        rewards = self._get_reward()
        terminated = self.step_count >= self.episode_length
        terminateds = {f"agent_{i}": terminated for i in range(self.num_agents)}
        truncateds = {f"agent_{i}": False for i in range(self.num_agents)}
        info = {
            "step": self.step_count,
            "imbalance": float(np.max(self.device_loads) - np.min(self.device_loads)),
            "loads": self.device_loads.copy()
        }
        return obs, rewards, terminateds, truncateds, info

    #返回每个 agent 的观测：当前负载 + 所有设备平均负载（2维特征）
    def _get_obs(self):
        avg_load = np.mean(self.device_loads)
        return {
            f"agent_{i}": np.array([self.device_loads[i], avg_load], dtype=np.float32)
            for i in range(self.num_agents)
        }

    #奖励机制设计：
        # imbalance = max - min：负载不均衡
        # penalty: 若某个设备负载超过0.9，惩罚
        # fairness: 各设备负载的标准差
        # base_reward = 1.0 - imbalance - penalty - fairness
    # 共享奖励模式：所有 agent 得到相同 reward
    # 非共享奖励：每个 agent 根据自身负载与平均值的偏差给予负值奖励（鼓励均衡）
    def _get_reward(self):
        imbalance = np.max(self.device_loads) - np.min(self.device_loads)
        penalty = np.sum(self.device_loads > 0.9) * 0.2
        fairness = np.std(self.device_loads)
        base_reward = 1.0 - imbalance - penalty - fairness
        if self.share_reward:
            return {f"agent_{i}": base_reward for i in range(self.num_agents)}
        else:
            avg = np.mean(self.device_loads)
            return {
                f"agent_{i}": -abs(self.device_loads[i] - avg)
                for i in range(self.num_agents)
            }

    #模拟节点随机下线（以 10% 概率选择一个节点 offline），离线后对应 active_mask = 0，负载清空为 0
    def _simulate_node_status(self):
        self.active_mask = np.ones(self.num_agents, dtype=np.float32)
        if np.random.rand() < 0.1:
            node = np.random.randint(self.num_agents)
            self.active_mask[node] = 0.0
            self.device_loads[node] = 0.0

    #随机给 12 个设备增加一定任务负载（0.050.2），模拟任务到达，capped 到最大 1.0
    def _generate_tasks(self):
        for _ in range(np.random.randint(1, 3)):
            i = np.random.randint(self.num_agents)
            load = np.random.uniform(0.05, 0.2)
            self.device_loads[i] += load
            self.device_loads[i] = min(self.device_loads[i], 1.0)

    #打印当前 step 的设备负载和在线状态
    def render(self, mode='human'):
        print(f"[Step {self.step_count}] Loads: {self.device_loads}, Active: {self.active_mask}")

    def close(self):
        pass
