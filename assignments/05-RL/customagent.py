import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(observation_space.shape[0], action_space.n).to(self.device)
        self.target_network = QNetwork(observation_space.shape[0], action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.last_observation = None
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        self.last_observation = observation
        if random.random() < self.epsilon:
            self.last_action = self.action_space.sample()
        else:
            observation = torch.FloatTensor(observation).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(observation)
            self.last_action = torch.argmax(q_values).item()
        return self.last_action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.last_observation is not None and self.last_action is not None:
            self.memory.append((self.last_observation, self.last_action, reward, observation, terminated))

        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminated = zip(*samples)

        batch_states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self.device)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        batch_terminated = torch.BoolTensor(batch_terminated).unsqueeze(1).to(self.device)

        q_values = self.q_network(batch_states)
        target_q_values = self.target_network(batch_next_states)
        q_values = q_values.gather(1, batch_actions)
        target_q_values, _ = target_q_values.max(1, keepdim=True)

        target_q_values = batch_rewards + (self.gamma * target_q_values * (1 - batch_terminated.float()))

        loss = self.loss_fn(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if np.random.random() < 0.005:
            self.target_network.load_state_dict(self.q_network.state_dict())
