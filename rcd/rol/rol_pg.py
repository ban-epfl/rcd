import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical

class ROLEnvironment(gym.Env):
    """
    Custom environment
    """

    def __init__(self, data, var_names, ci_test, goal=10, max_steps=20):
        super(ROLEnvironment, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)  # Move left or Move right
        # Example for using image as input:
        self.observation_space = spaces.Discrete(goal + 1)

        self.goal = goal
        self.state = 0  # Start at the left of the line
        self.max_steps = max_steps
        self.current_step = 0

        self.data = data
        self.var_names = var_names
        self.ci_test = ci_test

    def step(self, action):
        if action == 1:
            self.state += 1  # Move right
        elif action == 0:
            self.state -= 1  # Move left

        self.state = np.clip(self.state, 0, self.goal)
        self.current_step += 1

        # Reward is given for reaching the goal
        reward = 1 if self.state == self.goal else 0

        # Episode is done if the goal is reached or max steps exceeded
        done = self.state == self.goal or self.current_step >= self.max_steps

        # Optionally we can pass additional info, not used in this case
        info = {}

        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = 0
        self.current_step = 0
        return self.state  # return initial state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # Render the environment to the console
        line = '-' * self.goal
        print('Position on line: ' + line[:self.state] + "O" + line[self.state:])

    def close(self):
        pass


class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.forward(state)
        probability_distribution = Categorical(logits=logits)
        action = probability_distribution.sample()
        return action.item(), probability_distribution.log_prob(action)


class PolicyGradientAgent:
    def __init__(self, n_inputs, n_actions, hidden_size, learning_rate):
        self.policy_network = PolicyNetwork(n_inputs, n_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        action, log_prob = self.policy_network.act(state)
        self.saved_log_probs.append(log_prob)
        return action

    def train(self, gamma):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.saved_log_probs[:]
        del self.rewards[:]


# Create the environment and agent
env = gym.make('YourCustomEnv-v0')  # Replace with your custom environment
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_size = 128
learning_rate = 1e-2
gamma = 0.99  # discount factor

agent = PolicyGradientAgent(n_inputs, n_actions, hidden_size, learning_rate)

# Training loop
n_episodes = 1000
for episode in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        episode_reward += reward
    agent.train(gamma)
    print('Episode {}\tReward: {}'.format(episode, episode_reward))

# Close the environment
env.close()
