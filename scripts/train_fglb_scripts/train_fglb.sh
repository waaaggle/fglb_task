#!/bin/sh

env="LoadBalance"
scenario="default"
algo="mappo"
exp="test_run"
seed=1

CUDA_VISIBLE_DEVICES=0 python train_fglb.py \
--env_name ${env} \
--scenario_name ${scenario} \
--algorithm_name ${algo} \
--experiment_name ${exp} \
--seed ${seed} \
--num_agents 4 \
--episode_length 200 \
--share_reward \
--n_rollout_threads 8 \
--num_env_steps 1000000 \
--ppo_epoch 10 \
--num_mini_batch 2 \
--log_interval 10000 \
--save_interval 50000 \
--eval_interval 20000 \
--eval_episodes 10 \
--use_eval \
--user_name "waaaggle"
