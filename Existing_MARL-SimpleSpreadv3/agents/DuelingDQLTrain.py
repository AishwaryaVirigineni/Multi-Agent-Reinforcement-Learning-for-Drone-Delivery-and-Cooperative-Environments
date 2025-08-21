import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import time
import os
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import json
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, n_actions)
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        x = self.shared_fc(obs)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
class MultiAgentDuelingDQL:
    def __init__(self, max_steps=25, checkpoint_interval=100):
        self.max_steps = max_steps
        self.checkpoint_interval = checkpoint_interval
        self._configure_hyperparams()
        self._setup_criterions()
        self._setup_optimizers()
    def _configure_hyperparams(self):
        self.lr = 1e-3
        self.gamma = 0.9
        self.sync_target_every = 10
        self.replay_limit = 1000
        self.batch_size = 32
    def _setup_criterions(self):
        self.criteria = {f"agent_{i}": nn.MSELoss() for i in range(3)}
    def _setup_optimizers(self):
        self.optimizers = {f"agent_{i}": None for i in range(3)}
    def _init_replay_buffers(self):
        self.replay_buffers = {f"agent_{i}": ExperienceReplay(self.replay_limit) for i in range(3)}
    def _initialize_networks(self):
        obs_dim, action_dim = 18, 5
        agents = {}
        for i in range(3):
            agents[f"agent_{i}"] = DuelingQNetwork(obs_dim, action_dim)
        return agents
    def _sync_weights(self, agent_id):
        self.target_networks[agent_id].load_state_dict(self.policy_networks[agent_id].state_dict())
    def _assign_optimizer(self, agent_id):
        self.optimizers[agent_id] = torch.optim.Adam(
            self.policy_networks[agent_id].parameters(), lr=self.lr
        )
    def _select_action(self, agent_id, observations):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_spaces[agent_id].n)
        obs_tensor = torch.from_numpy(observations[agent_id]).float()
        with torch.no_grad():
            q_values = self.policy_networks[agent_id](obs_tensor)
        return torch.argmax(q_values).item()
    def _update_epsilon(self, current_episode, total_episodes):
        initial = 1.0
        final = 0.01
        decay_rate = (final / initial) ** (1.0 / total_episodes)
        self.epsilon = max(final, self.epsilon * decay_rate)
    def _save_model(self, agent_id, name_prefix, directory="./dueling_dqn_models/"):
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, f"{name_prefix}{agent_id}.pt")
        torch.save(self.policy_networks[agent_id].state_dict(), save_path)
    def _save_rewards(self, rewards, filename="rewards", directory="./dueling_dqn_models/"):
        os.makedirs(directory, exist_ok=True)
        agent_ids = sorted(rewards[0].keys())
        reward_array = np.array([[ep[aid] for aid in agent_ids] for ep in rewards])
        np.save(os.path.join(directory, f"{filename}"), reward_array)
    def train_agents(self, total_episodes):
        self.env = simple_spread_v3.parallel_env(local_ratio=0.5)
        agents = self.env.possible_agents
        self.env.reset()
        self._init_replay_buffers()
        self.policy_networks = self._initialize_networks()
        self.target_networks = self._initialize_networks()
        for agent_id in agents:
            self._sync_weights(agent_id)
            self._assign_optimizer(agent_id)
        all_rewards = []
        self.epsilon = 1.0
        self.epsilon_history = []
        for ep in range(total_episodes):
            start_time = time.time()
            observations = self.env.reset()[0]
            agent_actions = {aid: 0 for aid in agents}
            episode_reward = {aid: 0 for aid in agents}
            for t in range(self.max_steps):
                for aid in agents:
                    agent_actions[aid] = self._select_action(aid, observations)
                next_obs, reward, done, _, _ = self.env.step(agent_actions)
                for aid in agents:
                    transition = (
                        observations[aid],
                        agent_actions[aid],
                        next_obs[aid],
                        reward[aid],
                        done[aid]
                    )
                    episode_reward[aid] += reward[aid] / self.max_steps
                    self.replay_buffers[aid].push(transition)
                observations = next_obs
            for aid in agents:
                if len(self.replay_buffers[aid]) >= self.batch_size:
                    batch = self.replay_buffers[aid].sample_batch(self.batch_size)
                    self._learn_from_batch(batch, aid)
                    self._update_epsilon(ep, total_episodes)
                    self._sync_weights(aid)
            self.epsilon_history.append(self.epsilon)
            elapsed = round(time.time() - start_time, 2)
            self._print_episode_summary(ep, elapsed, episode_reward)
            all_rewards.append(episode_reward)
            if ep % self.checkpoint_interval == 0 and ep != 0:
                self._save_rewards(all_rewards, filename=f"ep{ep}_rewards.npy")
                for aid in agents:
                    self._save_model(aid, f"ep{ep}_")
        for aid in agents:
            self._save_model(aid, "final_")
        self._save_rewards(all_rewards, filename="final_rewards.npy")
        np.save(os.path.join("./dueling_dqn_models/", "epsilon_history.npy"), self.epsilon_history)
        self.env.close()
        return all_rewards
    def _learn_from_batch(self, batch, agent_id):
        policy_net = self.policy_networks[agent_id]
        target_net = self.target_networks[agent_id]
        q_preds, q_targets = [], []
        for s, a, s_next, r, _ in batch:
            state_tensor = torch.FloatTensor(s)
            next_state_tensor = torch.FloatTensor(s_next)
            with torch.no_grad():
                next_q_values = policy_net(next_state_tensor)
                next_action = torch.argmax(next_q_values).item()
                target_q_value = target_net(next_state_tensor)[next_action]
                q_target = r + self.gamma * target_q_value
            q_values = policy_net(state_tensor)
            q_value_copy = q_values.clone()
            q_value_copy[a] = q_target
            q_preds.append(q_values)
            q_targets.append(q_value_copy)
        loss = self.criteria[agent_id](torch.stack(q_preds), torch.stack(q_targets))
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
    def _print_episode_summary(self, episode_num, duration, rewards):
        print(f"Episode {episode_num} ----  Time: {duration}s")
        print("Average Rewards:", {k: round(v, 2) for k, v in rewards.items()})
        print()