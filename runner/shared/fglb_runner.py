import numpy as np
from fglb_task.runner.shared.base_runner import Runner

class FglbRunner(Runner):
    def __init__(self, config, envs, eval_envs, run_dir):
        super(FglbRunner, self).__init__(config, envs, eval_envs, run_dir)

    def run(self):
        self.warmup()
        for episode in range(self.episodes):
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            self.compute()
            self.train()

            if self.use_eval and self.total_env_steps % self.eval_interval < self.episode_length:
                self.eval(self.total_env_steps)
            if self.total_env_steps % self.save_interval < self.episode_length:
                self.save()

            self.total_env_steps += self.episode_length * self.n_rollout_threads

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = 0.0
        active_masks = np.copy(masks)
        self.buffer.insert(obs, obs, rnn_states, rnn_states_critic, actions,
                           action_log_probs, values, rewards, masks, active_masks, masks)

    def eval(self, total_num_steps):
        eval_episode = 0
        eval_rewards = []
        obs = self.eval_envs.reset()
        rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                               self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while eval_episode < self.eval_episodes:
            actions, _, rnn_states = self.policy.act(obs, rnn_states, masks, deterministic=True)
            obs, rewards, dones, infos = self.eval_envs.step(actions)
            eval_rewards.append(np.mean(rewards))
            for done in dones:
                if done:
                    eval_episode += 1

        avg_reward = np.mean(eval_rewards)
        print(f"[Eval] Step: {total_num_steps} | Avg Reward: {avg_reward:.3f}")
        if self.use_wandb:
            self.wandb.log({"eval/avg_reward": avg_reward}, step=total_num_steps)
        else:
            self.writter.add_scalar("eval/avg_reward", avg_reward, total_num_steps)
