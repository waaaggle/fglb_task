"""
定义了一个自定义环境包装类 FootballEnv，用于将 Google Research Football 环境（gfootball） 封装成多智能体强化学习（MARL）框架可以使用的形式。
该包装器主要做了以下几件事：
创建并管理 Google Football 环境；
将单智能体或多智能体接口统一为 MARL 可接受的标准格式；
处理 observation、reward、info 等；
提供环境的 reset()、step()、close() 接口；
实现动作空间、观测空间的封装；
实现共享奖励机制等功能。
"""
import random

import gfootball.env as football_env
from gym import spaces
import numpy as np


class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        #从 args 中读取关键设置，比如智能体数量和使用的场景。
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        
        # make env创建 Google Football 环境
        # 根据是否需要渲染或保存视频的需求，调用gfootball提供的 create_environment()来创建底层环境。常用参数说明：
        # representation：如'raw'、'simple115v2'、'extracted'、'pixels'；
        # stacked：是否叠加多个时间帧；
        # number_of_left_players_agent_controls：左队（智能体）数量；
        # render：是否渲染；
        # write_video：是否保存视频；
        # logdir：保存视频的目录。
        if not (args.use_render and args.save_videos):
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                render=(args.use_render and args.save_gifs)
            )
        else:
            # render env and save videos
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                # video related params
                write_full_episode_dumps=True,
                render=True,
                write_video=True,
                dump_frequency=1,
                logdir=args.video_dir
            )
            
        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        #设置空间属性（动作空间 / 观测空间）
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        #是否单智能体
        if self.num_agents == 1:
            #直接使用底层环境的 action/obs 空间。
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
        else:
            #对每个智能体分别设置空间；
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                #share_observation_space 是用于 centralized critic 时共享状态的空间（这里和 observation_space 是相同结构）。
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))
                self.share_observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))

    #调用底层环境重置；
    def reset(self):
        obs = self.env.reset()
        #用 _obs_wrapper 处理 observation 形状（比如在单智能体下添加 batch 维度）。
        obs = self._obs_wrapper(obs)
        return obs

    #执行一步动作
    def step(self, action):
        #执行环境一步，获得观察、奖励、结束标志、信息。
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        #奖励 reshape 并支持共享奖励
        reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            #则每个智能体都获得全局总奖励（用于协作训练）。
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.num_agents
        #将 done 标志扩展为每个智能体一份；
        done = np.array([done] * self.num_agents)
        #用 _info_wrapper() 增强 info 内容。
        info = self._info_wrapper(info)
        return obs, reward, done, info

    #控制随机性；
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    #清理资源。
    def close(self):
        self.env.close()

    # 单智能体时增加维度，返回形状为[1, obs_dim]；
    # 多智能体时直接返回[num_agents, obs_dim]。
    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    #提供更多关于环境和智能体的诊断信息
    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        # max_steps：本场比赛的最大步数；
        # active：每个智能体是否参与；
        # designated：每个智能体是否是被控制者；
        # sticky_actions：动作粘性，用于表示动作重复状态。
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info
