import numpy as np
from fglb_task.envs.fglb_dut.gym_fglb_env import GymFglbEnv

class FglbWrapper:
    def __init__(self, args):
        self.env = GymFglbEnv(
            num_agents=args.num_agents,
            episode_length=args.episode_length,
            share_reward=args.share_reward,
            dynamic=True
        )
        self.num_agents = args.num_agents
        self.obs_dim = 2
        self.share_obs_dim = self.num_agents * self.obs_dim

    def reset(self):
        obs_dict, _ = self.env.reset()
        obs = self._dict_to_array(obs_dict)
        return obs, self._get_share_obs(obs)

    def step(self, actions):
        action_dict = {f"agent_{i}": actions[i] for i in range(self.num_agents)}
        obs_dict, reward_dict, dones, truncs, info = self.env.step(action_dict)
        obs = self._dict_to_array(obs_dict)
        rewards = self._dict_to_array(reward_dict, expand_dim=True)
        done_flag = list(dones.values())[0]
        return obs, rewards, self._get_share_obs(obs), done_flag, info

    def _dict_to_array(self, d: dict, expand_dim=False):
        arr = [d[f"agent_{i}"] for i in range(self.num_agents)]
        arr = np.stack(arr)
        if expand_dim:
            arr = arr[:, np.newaxis]
        return arr

    def _get_share_obs(self, obs):
        return np.tile(obs.flatten(), (self.num_agents, 1))

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
