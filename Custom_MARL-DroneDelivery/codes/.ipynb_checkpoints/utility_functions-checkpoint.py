import pickle
import matplotlib.pyplot as plt
import numpy as np
from DroneDelivery_Env import DroneDeliveryMultiAgentEnv
from QLearning import QLearningAgent
from SARSALearning import SARSAAgent
from DoubleQLearning import DoubleQLearningAgent

def load_agents(env, prefix="qlearning"):
    agents = {}
    for aid in env.agent_ids:
        with open(f"{prefix}_{aid}_q_table.pkl", "rb") as f:
            q_data = pickle.load(f)

        if prefix.lower() == "doubleqlearning":
            agent = DoubleQLearningAgent(aid, env)
            agent.q_table_a, agent.q_table_b = q_data
        elif prefix.lower() == "sarsa":
            agent = SARSAAgent(aid, env)
            agent.q_table = q_data
        else:
            agent = QLearningAgent(aid, env)
            agent.q_table = q_data

        agent.epsilon = 0.0  # Force greedy
        agents[aid] = agent
    return agents

def load_training_metrics(prefix="qlearning"):
    with open(f"{prefix}_total_rewards.pkl", "rb") as f:
        total_rewards = pickle.load(f)
    with open(f"{prefix}_epsilons.pkl", "rb") as f:
        epsilons = pickle.load(f)
    return total_rewards, epsilons

def plot_training_results(total_rewards, epsilons):
    episodes = range(len(total_rewards))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(episodes, total_rewards)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Total Reward per Episode')
    axs[0].grid(True)

    axs[1].plot(episodes, epsilons)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Epsilon Value')
    axs[1].set_title('Epsilon Decay over Episodes')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_greedy(env, agents, episodes=10):
    rewards_per_episode = []

    for ep in range(episodes):
        observations, _ = env.reset()
        done = {aid: False for aid in env.agent_ids}
        total_reward = 0

        while not all(done.values()):
            actions = {}
            for aid, agent in agents.items():
                x, y, carrying = observations[aid]
                if hasattr(agent, 'q_table_a') and hasattr(agent, 'q_table_b'):
                    combined_q = agent.q_table_a[x, y, carrying] + agent.q_table_b[x, y, carrying]
                    actions[aid] = np.argmax(combined_q)
                else:
                    actions[aid] = np.argmax(agent.q_table[x, y, carrying])

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(rewards.values())
            observations = next_observations
            done = {aid: terminations[aid] or truncations[aid] for aid in env.agent_ids}

        rewards_per_episode.append(total_reward)
        print(f"Greedy Evaluation Episode {ep+1}: Total Reward = {total_reward}")

    plt.figure()
    plt.plot(range(episodes), rewards_per_episode, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Greedy Policy Total Reward per Episode')
    plt.grid(True)
    plt.show()

def render_single_greedy_episode(env, agents):
    observations, _ = env.reset()
    done = {aid: False for aid in env.agent_ids}

    while not all(done.values()):
        actions = {}
        for aid, agent in agents.items():
            x, y, carrying = observations[aid]
            if hasattr(agent, 'q_table_a') and hasattr(agent, 'q_table_b'):
                combined_q = agent.q_table_a[x, y, carrying] + agent.q_table_b[x, y, carrying]
                actions[aid] = np.argmax(combined_q)
            else:
                actions[aid] = np.argmax(agent.q_table[x, y, carrying])

        next_observations, rewards, terminations, truncations, _ = env.step(actions)

        # Print detailed step-by-step info
        for aid in env.agent_ids:
            print(f"{aid} ➔ Action: {actions[aid]}, Local Obs: {observations[aid]}, Reward: {rewards[aid]}, Terminated: {terminations[aid]}, Truncated: {truncations[aid]}")

        env.render(plot=True)
        observations = next_observations
        done = {aid: terminations[aid] or truncations[aid] for aid in env.agent_ids}