import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OneDLineEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, goal=10, max_steps=20):
        super(OneDLineEnv, self).__init__()

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


# Testing the environment
if __name__ == "__main__":
    env = OneDLineEnv()
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        env.render()

    env.close()
