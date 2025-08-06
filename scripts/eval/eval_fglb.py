# file: plot_eval.py
import matplotlib.pyplot as plt
import json
import os

def plot_eval_curve(log_file):
    steps = []
    rewards = []

    with open(log_file, 'r') as f:
        for line in f:
            if '"eval/avg_reward"' in line:
                data = json.loads(line)
                steps.append(data['step'])
                rewards.append(data['eval/avg_reward'])

    plt.figure(figsize=(8,5))
    plt.plot(steps, rewards, marker='o')
    plt.title("Eval Reward Curve")
    plt.xlabel("Total Env Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_file = "./wandb/latest-run/files/output.log"  # 或者你自定义的日志文件
    plot_eval_curve(log_file)
