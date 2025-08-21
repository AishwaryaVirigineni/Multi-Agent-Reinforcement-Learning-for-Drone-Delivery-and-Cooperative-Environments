import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import os
def load_model(agent_class, agent_id, model_dir):
    model_path = os.path.join(model_dir, f"final_{agent_id}.pt")
    model = agent_class(18, 5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def load_rewards(reward_dir):
    reward_path = os.path.join(reward_dir, "final_rewards.npy")
    return np.load(reward_path)
def load_epsilons(model_dir):
    epsilon_path = os.path.join(model_dir, "epsilon_history.npy")
    return np.load(epsilon_path)
def plot_agent_rewards(rewards, agent_index=None, window=100):
    episodes = np.arange(len(rewards))
    smoothed = lambda x: np.convolve(x, np.ones(window)/window, mode='valid')
    if agent_index is not None:
        raw = rewards[:, agent_index]
        plt.plot(episodes, raw, label=f"Agent {agent_index} Reward", alpha=0.4)
        plt.plot(episodes[window-1:], smoothed(raw), '--', label=f"Agent {agent_index} Moving Avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward Curve for Agent {agent_index}")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for i in range(3):
            raw = rewards[:, i]
            axs[i].plot(episodes, raw, label=f"Agent {i} Reward")
            axs[i].plot(episodes[window-1:], smoothed(raw), '--', label=f"Agent {i} Moving Avg")
            axs[i].set_title(f"Agent {i} Reward")
            axs[i].set_xlabel("Episode")
            axs[i].set_ylabel("Reward")
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()
def plot_avg_rewards(rewards, window = 100):
    window=100
    mean_rewards = rewards.mean(axis=1)
    smoothed = lambda x: np.convolve(x, np.ones(window)/window, mode='valid')
    episodes = np.arange(len(mean_rewards))
    plt.plot(episodes, mean_rewards, label="Average Reward (All Agents)")
    plt.plot(episodes[window-1:], smoothed(mean_rewards), '--', label=f"Agent Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Across Agents")
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_epsilon_decay(epsilons):
    episodes = np.arange(len(epsilons))
    plt.plot(episodes, epsilons, color='blue', label='Epsilon')
    plt.xlabel("Episode")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()
def evaluate_policy(models, num_episodes=10):
    env = simple_spread_v3.parallel_env(local_ratio=0.5)
    env.reset()
    total_rewards = {f"agent_{i}": 0.0 for i in range(3)}
    per_episode_avg = []
    for ep in range(num_episodes):
        observations = env.reset()[0]
        done = {aid: False for aid in env.possible_agents}
        episode_reward = {aid: 0.0 for aid in env.possible_agents}
        while not all(done.values()):
            actions = {}
            for aid in env.possible_agents:
                obs_tensor = torch.from_numpy(observations[aid]).float()
                with torch.no_grad():
                    q_vals = models[aid](obs_tensor)
                    actions[aid] = torch.argmax(q_vals).item()
            observations, rewards, done, _, _ = env.step(actions)
            for aid in rewards:
                episode_reward[aid] += rewards[aid]
        for aid in env.possible_agents:
            total_rewards[aid] += episode_reward[aid]
        avg_episode_reward = np.mean(list(episode_reward.values()))
        per_episode_avg.append(avg_episode_reward)
    avg_rewards = {aid: round(r / num_episodes, 2) for aid, r in total_rewards.items()}
    print("Greedy Evaluation Results (Avg over 10 episodes):", avg_rewards)
    env.close()
    return avg_rewards, per_episode_avg
    avg_rewards = {aid: round(r / num_episodes, 2) for aid, r in total_rewards.items()}
    print("Greedy Evaluation Results (Avg over 10 episodes):", avg_rewards)
    env.close()
    return avg_rewards
def plot_greedy_rewards_curve(per_episode_avg):
    plt.plot(range(len(per_episode_avg)), per_episode_avg, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.title("Greedy Evaluation Curve (10 Episodes)")
    plt.grid(True)
    plt.show()
def render_greedy_episode(models):
    env = simple_spread_v3.parallel_env(local_ratio=0.5, continuous_actions=False, render_mode="human")
    env.reset()
    env.render()
    observations = env.reset()[0]
    done = {aid: False for aid in env.possible_agents}
    while not all(done.values()):
        actions = {}
        for aid in env.possible_agents:
            obs_tensor = torch.from_numpy(observations[aid]).float()
            with torch.no_grad():
                q_vals = models[aid](obs_tensor)
                actions[aid] = torch.argmax(q_vals).item()
        observations, _, done, _, _ = env.step(actions)
        env.render()
    env.close()