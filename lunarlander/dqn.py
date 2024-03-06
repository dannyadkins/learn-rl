import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000
max_steps = 1000

env = gym.make('LunarLander-v2', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = DQN(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # env.render()  
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        # Take the action and observe the next state and reward
        next_state, reward, done, _, _= env.step(action)
        total_reward += reward

        # Update the Q-Network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = q_network(state_tensor)
        next_q_values = q_network(next_state_tensor)
        target_q_value = reward + gamma * torch.max(next_q_values)
        loss = F.mse_loss(q_values[0, action], target_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

env.close()