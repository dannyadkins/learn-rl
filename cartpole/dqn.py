import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import os

# Parse CLI arguments
parser = argparse.ArgumentParser(description='DQN Agent')
parser.add_argument('--save', type=str, help='Path to save the model')
parser.add_argument('--load', type=str, help='Path to load the model')
args = parser.parse_args()

# Hyperparameters
EPISODES = 500
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.tensor(actions).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float()
        rewards = torch.tensor(rewards).unsqueeze(1)
        dones = torch.tensor(dones).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (~dones) * GAMMA * next_q_values
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

# training loop
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)

if args.load:
    if os.path.isfile(args.load):
        agent.load(args.load)
        print(f"Model loaded from {args.load}")
    else:
        print(f"No model found at {args.load}, starting training from scratch.")

rewards = []
for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, next_state, reward, done)
        agent.replay()
        state = next_state
        total_reward += reward

    agent.update_epsilon()
    rewards.append(total_reward)
    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if args.save:
    agent.save(args.save)
    print(f"Model saved to {args.save}")

env.close()

# Plotting the rewards
plt.plot(rewards)
plt.title('Reward over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()