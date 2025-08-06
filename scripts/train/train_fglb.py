import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, r'D:\llm_practise\RL')
# print(sys.path, os.listdir(sys.path[0]))

from fglb_task.config import get_config
from fglb_task.envs.fglb_dut.make_env import make_fglb_env
from fglb_task.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from fglb_task.runner.shared.fglb_runner import FglbRunner

def make_train_env(all_args):
    env_fns = [make_fglb_env(all_args, i) for i in range(all_args.n_rollout_threads)]
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv(env_fns)
    else:
        return ShareSubprocVecEnv(env_fns)

if __name__ == "__main__":
    args = get_config()
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device("cuda:0" if args.cuda else "cpu")
    args.device = device

    # 保存路径
    run_dir = Path(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    #构造环境
    envs = make_train_env(args)
    eval_envs = make_train_env(args) if args.use_eval else None

    #实例化模型，初始化模型train函数
    runner = FglbRunner(args, envs, eval_envs, run_dir)
    # 探索环境, 调用模型train函数
    runner.run()

