import numpy as np
import pickle
from DroneDelivery_Env import DroneDeliveryMultiAgentEnv
class DoubleQLearningAgent:
    def __init__(self, agent_id, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay_factor=0.995):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_factor = decay_factor
        self.env = env
        self.q_table_a = np.zeros((env.grid_size, env.grid_size, 2, env.action_space.n))
        self.q_table_b = np.zeros((env.grid_size, env.grid_size, 2, env.action_space.n))
    def select_action(self, obs):
        x, y, carrying = obs
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            combined_q = self.q_table_a[x, y, carrying] + self.q_table_b[x, y, carrying]
            return np.argmax(combined_q)
    def update(self, obs, action, reward, next_obs, done):
        x, y, carrying = obs
        nx, ny, ncarrying = next_obs
        if np.random.rand() < 0.5:
            best_action = np.argmax(self.q_table_a[nx, ny, ncarrying])
            td_target = reward + self.gamma * self.q_table_b[nx, ny, ncarrying, best_action] * (1 - done)
            td_error = td_target - self.q_table_a[x, y, carrying, action]
            self.q_table_a[x, y, carrying, action] += self.alpha * td_error
        else:
            best_action = np.argmax(self.q_table_b[nx, ny, ncarrying])
            td_target = reward + self.gamma * self.q_table_a[nx, ny, ncarrying, best_action] * (1 - done)
            td_error = td_target - self.q_table_b[x, y, carrying, action]
            self.q_table_b[x, y, carrying, action] += self.alpha * td_error
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.q_table_a, self.q_table_b), f)
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table_a, self.q_table_b = pickle.load(f)
def train_double_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, max_timesteps=1000):
    decay_factor = 0.995
    agents = {aid: DoubleQLearningAgent(aid, env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, decay_factor=decay_factor) for aid in env.agent_ids}
    total_rewards = []
    epsilons = []
    for ep in range(episodes):
        observations, _ = env.reset()
        done = {aid: False for aid in env.agent_ids}
        timesteps = 0
        episode_reward = 0
        while not all(done.values()) and timesteps < max_timesteps:
            actions = {aid: agent.select_action(observations[aid]) for aid, agent in agents.items()}
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            for aid, agent in agents.items():
                agent.update(observations[aid], actions[aid], rewards[aid], next_observations[aid], terminations[aid] or truncations[aid])
            observations = next_observations
            done = {aid: terminations[aid] or truncations[aid] for aid in env.agent_ids}
            timesteps += 1
            episode_reward += sum(rewards.values())
        total_rewards.append(episode_reward)
        epsilons.append(next(iter(agents.values())).epsilon)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} completed")
    for aid, agent in agents.items():
        agent.save(f"doubleqlearning_{aid}_q_table.pkl")
    with open("doubleqlearning_total_rewards.pkl", "wb") as f:
        pickle.dump(total_rewards, f)
    with open("doubleqlearning_epsilons.pkl", "wb") as f:
        pickle.dump(epsilons, f)
    return agents, total_rewards, epsilons