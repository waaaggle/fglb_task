import torch
import numpy as np
from fglb_task.envs.fglb_dut.gym_fglb_env import GymFglbEnv
from fglb_task.envs.fglb_dut.fglb_wrapper import FglbWrapper
from fglb_task.algorithms.r_mappo.r_mappo_policy import R_MAPPOPolicy
from fglb_task.config import get_config

def load_model(model_path, policy):
    policy.actor.load_state_dict(torch.load(model_path))
    policy.actor.eval()

def deploy():
    args = get_config()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_agents = 4
    args.share_policy = True

    # 初始化环境
    env = FglbWrapper(args)
    obs, share_obs = env.reset()
    rnn_states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((args.num_agents, 1), dtype=np.float32)

    # 加载模型策略
    policy = R_MAPPOPolicy(args, obs_space=obs.shape[-1], share_obs_space=share_obs.shape[-1],
                           act_space=env.env.env.action_space["agent_0"], device=args.device)
    load_model("./trained_model.pt", policy)

    done = False
    step = 0
    while not done and step < 200:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
            share_obs_tensor = torch.tensor(share_obs, dtype=torch.float32, device=args.device)
            rnn_states_tensor = torch.tensor(rnn_states, dtype=torch.float32, device=args.device)
            masks_tensor = torch.tensor(masks, dtype=torch.float32, device=args.device)

            values, actions, action_log_probs, rnn_states_tensor, _ = policy.get_actions(
                share_obs_tensor, obs_tensor, rnn_states_tensor, rnn_states_tensor, masks_tensor
            )
            actions = actions.cpu().numpy()

        obs, rewards, share_obs, done, info = env.step(actions)
        rnn_states = rnn_states_tensor.cpu().numpy()
        env.render()
        step += 1

if __name__ == "__main__":
    deploy()
