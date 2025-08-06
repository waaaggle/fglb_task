"""
Modified from OpenAI Baselines code to work with multi-agent envs
这段代码是基于 OpenAI Baselines 的向量化环境（Vectorized Environment）代码，
经过修改以支持多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）环境。
它通过 Python 的 multiprocessing 模块实现多个环境的并行运行，适用于加速强化学习训练的场景。
代码的核心是定义了一组类，用于管理多个环境的并行执行，支持单智能体和多智能体环境，
提供共享观测空间（share_observation_space）和动作空间（action_space）的功能。
以下是对代码的详细中文解释，涵盖其结构、功能和关键实现细节。
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from fglb_task.utils.util import tile_images

#CloudpickleWrapper 是一个辅助类，用于序列化环境创建函数（env_fn），以便在 multiprocessing 中正确传递复杂对象。
#这是一个对象包装器，专门为多进程通信（multiprocessing）设计。Python 的多进程默认用 pickle 序列化对象，但有些环境创建函数（如 lambda、局部函数、Gym 环境等）不能被 pickle 正常序列化。
# 这时用 cloudpickle 能解决更多类型的序列化问题。把环境构造函数（env_fn）包装起来传递给子进程，保证不会因为序列化问题报错。
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x          #存储对象 x，通常是一个环境函数或实例

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x) #在对象被发送到子进程时，使用 cloudpickle.dumps() 序列化 self.x

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)  #反序列化对象，使用 pickle.loads()（这里似乎有不一致，理想情况下应使用 cloudpickle.loads() 以保持对称性）

#ShareVecEnv 是一个抽象基类，定义了向量化环境的接口，适用于批量处理多个环境的数据。定义了向量化环境（即批量管理多个环境实例）的抽象接口。
class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    # num_envs：并行运行的环境数量。
    # observation_space：每个环境的观测空间（Gym风格）。
    # share_observation_space：多智能体环境中的共享观测空间，包含所有智能体的全局状态信息。
    # action_space：动作空间，定义智能体可执行的动作。
    # closed和viewer：用于管理环境关闭状态和渲染窗口。
    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    #重置所有环境，返回初始观测。
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    #异步发送动作给所有环境（准备执行）。
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    #等待所有环境执行完 step，收集数据（观测、奖励、done、info）。
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass
    #关闭所有环境，释放资源
    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    #关闭所有环境与资源。
    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    #同步执行一步（兼容老接口）。
    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    #渲染所有环境，拼接成一张图像（支持 human 或 rgb_array 模式）。
    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    #获取每个环境的渲染图像。
    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    #解包，获取最原始的环境实例。
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self
    #获取或创建显示窗口
    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

# worker 适配的是标准单智能体环境（如 OpenAI Gym 环境）
#这是每个子进程里实际控制一个环境实例的主循环。每个 worker 只负责一个环境。
#多进程原理：主进程和每个 worker 进程通过 Pipe（管道）通信，主进程批量发命令，worker 并行处理，提升采样效率。
# remote：一个 pipe，用来和主进程通信（接收命令，返回结果）。
# parent_remote.close()：关闭父端，只保留子端通信。
# env = env_fn_wrapper.x()：通过 CloudpickleWrapper 解包创建环境实例。
#运行在每个子进程中，负责管理单个环境实例，并与主进程通信。
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        #接收主进程通过 Pipe 发送的命令（cmd）和数据（data），如 step、reset、render、close 等。
        cmd, data = remote.recv()
        if cmd == 'step':
            #执行一步动作（data 是动作），返回观测、奖励、done、info。若环境结束（done），则自动reset。
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()  #重置环境，返回初始观测。
            else:
                if np.all(done):
                    ob = env.reset()  #重置环境，返回初始观测。
            remote.send((ob, reward, done, info))  #将结果发送回主进程。
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))  #将结果发送回主进程。
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)  #渲染环境，返回图像（或直接显示）。
                remote.send(fr)  #将结果发送回主进程。
            elif data == "human":
                env.render(mode=data)    #渲染环境，返回图像（或直接显示）。
        elif cmd == 'reset_task':
            ob = env.reset_task()     #重置任务（有些环境有这个接口）
            remote.send(ob)  #将结果发送回主进程。
        elif cmd == 'close':
            env.close()  #关闭环境与通信，退出循环（break）。
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))  #将结果发送回主进程。
        else:
            raise NotImplementedError

#GuardSubprocVecEnv：p.daemon = False，主进程崩溃后子进程不自动终止（便于调试），但可能导致僵尸进程。
#SubprocVecEnv：p.daemon = True，主进程退出时子进程自动终止，更安全。
#ShareVecEnv 的具体实现，使用多进程并行运行多个 Gym 环境
class GuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        #创建 nenvs 对 Pipe，分别用于主进程与子进程通信，得到多个pipe元组队，然后再zip合并
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        #启动 nenvs 个子进程，每个进程运行 worker，管理一个环境实例。args为每个子进程的参数，env_fn为环境创建函数
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        #启动子进程
        for p in self.ps:
            #设置非守护进程属性，如果不设置，则主进程退出子进程自动终止，
            # 必须显式调用 p.join() 或 p.terminate() 来关闭，否则可能导致僵尸进程（zombie process）。
            p.daemon = False  # could cause zombie process，
            p.start()
        #主进程关闭 work_remotes 端，仅保留 remotes 端
        for remote in self.work_remotes:
            remote.close()
        #通过第一个 remote 获取环境的观测空间、共享观测空间和动作空间，调用父类 ShareVecEnv 的构造函数初始化元数据。
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        #调用父类构造函数，初始化环境数量等元数据。实际上保存数据
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    #依次向每个 remote（即每个子进程）发送 "step" 命令和对应动作。
    # 不等待结果，直接返回。此时子进程开始并行执行 step。
    def step_async(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    #从每个 remote 阻塞接收 step 的返回结果（各进程会并发采样，主进程收集全部结果后合并）。
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        #zip(*results) 将每个环境的返回按字段分组（观测、奖励、done、info）。
        obs, rews, dones, infos = zip(*results)
        #np.stack 合并成批量张量，返回给上层算法。infos 一般是字典列表，直接返回。
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    #批量向所有环境发送 reset 命令。
    #收集所有环境的初始观测，合成批量张量返回。
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    #一些环境支持 reset_task（比如多任务学习），用法与 reset 类似。
    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    # 如果环境已关闭，直接返回。如果上次 step_async 后还没收集结果，先收集一次，防止死锁。
    # 向所有子进程发送'close'命令，并等待所有进程退出。标记为 closed。
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

#GuardSubprocVecEnv：p.daemon = False，主进程崩溃后子进程不自动终止（便于调试），但可能导致僵尸进程。
#SubprocVecEnv：p.daemon = True，主进程退出时子进程自动终止，更安全。
class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    #批量让所有子进程渲染当前画面。
    #如果是 rgb_array 模式，收集所有环境的图片，合成批量图片返回。
    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 

#shareworker 适配的是多智能体或带全局观测/可用动作的环境
#适合用在多智能体环境、MARL算法（如 MAPPO、MAVEN、MADDPG）等需要全局观测、可用动作信息的场景。
def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError
# 配套worker：shareworker
# 典型接口：
# step 返回：obs, share_obs, reward, done, info, available_actions
# reset 返回：obs, share_obs, available_actions
# 适合多智能体环境，如MAPPO、SMAC等——既有个体观测(obs)，也有全局观测(share_obs)和可用动作(available_actions)
# reset 不带参数
# 适用场景：多智能体、多信息（全局观测/可用动作）的环境采样
class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

#	简单或单智能体
# env.step(data)返回4个值：ob, reward, done, info
# env.reset(data)返回1个值：ob
# 没有全局观测、没有可用动作、没有多智能体相关内容
# 适合单智能体或者接口很简单的环境，但reset可以带参数（如reset选择某种初始状态）
def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

# 配套worker：choosesimpleworker
# 典型接口：
# step 返回：obs, reward, done, info
# reset 返回：obs
# reset 需要外部传入一个参数（如reset_choose），批量每个环境可不同。
# 没有全局观测和可用动作
# 适用场景：单智能体或简单多智能体环境，只需要观测、奖励、done、info，且reset时可指定初始状态。
class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
#chooseworker是多智能体/复杂环境专用，能返回全局观测和可用动作。
# env.step(data)返回6个值：ob, s_ob, reward, done, info, available_actions
# available_actions：当前可用动作（多智能体/部分动作受限环境专用）
# env.reset(data)返回3个值：ob, s_ob, available_actions
# 适合多智能体环境，如MARL、MAPPO、SMAC等，其中需要全局观测和可用动作的场景
# 采样时可以直接获得所有智能体的局部观测、全局状态以及可用动作mask
def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

# 配套worker：chooseworker
# 典型接口：
# step 返回：obs, share_obs, reward, done, info, available_actions
# reset 返回：obs, share_obs, available_actions
# reset 要传入选择参数（如reset_choose）
# 兼容多智能体、全局观测、可用动作，且reset可指定初始状态
# 适用场景：复杂多智能体环境，既有全局观测、可用动作，又需要reset时指定环境初始状态。
class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# 和choosesimpleworker完全一样，接口一致。
# 可能只是命名上区分用途（如“守卫”环境或特殊实验），但接口不变。
# 还是4返回值step、1返回值reset，reset也可带参数。
def chooseguardworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

# 配套worker：chooseguardworker
# 典型接口：
# step 返回：obs, reward, done, info
# reset 返回：obs
# reset 需要外部传入一个参数（如reset_choose）
# 没有全局观测和可用动作
# 和ChooseSimpleSubprocVecEnv接口一模一样，但通常是为了特定实验命名区分（如“守卫”/“攻击者”等）
# 适用场景：与ChooseSimpleSubprocVecEnv类似，主要是命名用于区分不同实验/角色/环境类型。
class ChooseGuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseguardworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
# 适配环境：标准单智能体或无全局观测/可用动作的多智能体环境
# step返回：observation, reward, done, info
# reset返回：observation
# reset不带参数
# step_wait：
# 采样后如发现done为True，会自动reset该环境（只更新obs）没有share_obs和available_actions
# 适用场景：最常见的Gym单智能体环境，也可以用于多智能体但不需要全局观测和可用动作的场景。
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

# 适配环境：多智能体，返回全局观测和可用动作的环境（如MAPPO/SMAC）
# step返回：observation, share_obs, reward, done, info, available_actions
# reset返回：observation, share_obs, available_actions
# reset不带参数
# step_wait：
# 采样后如发现done为True，会自动reset该环境，并同时更新obs、share_obs和available_actions专为多智能体/全局观测/可用动作设计
# 适用场景：如SMAC、MAPPO等多智能体强化学习环境，需要全局观测和动作mask。
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

# 适配环境：多智能体，返回全局观测和可用动作，同时reset可指定初始状态
# step返回：observation, share_obs, reward, done, info, available_actions
# reset返回：observation, share_obs, available_actions
# reset带参数（如reset_choose，批量可以每个环境不同，常用于指定初始位置/任务等）
# step_wait：
# 不自动reset（仅收集step结果，reset由上层决定何时调用）对reset的参数支持是最大区别
# 适用场景：如多任务/多场景环境，reset时可选择不同的初始状态（如不同地图、位置、任务）。
class ChooseDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

# 适配环境：单智能体或无全局观测/可用动作，reset可指定初始状态
# step返回：observation, reward, done, info
# reset返回：observation
# reset带参数（如reset_choose）
# 没有share_obs和available_actions
# step_wait：只收集step结果，不自动reset
# 适用场景：单智能体或简单多智能体环境，需要reset时支持指定初始状态。
class ChooseSimpleDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        obs = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
