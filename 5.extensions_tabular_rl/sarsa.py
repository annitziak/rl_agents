import gymnasium as gym
from tqdm import tqdm

from rl2025.constants import EX2_QL_CONSTANTS as CONSTANTS
from rl2025.exercise2.utils import evaluate

from rl2025.exercise2.agents import Agent
from rl2025.exercise2.agents import QLearningAgent

import random


class SARSAAgent(Agent):
    """Agent using the SARSA algorithm for training"""

    def __init__(self, alpha: float, **kwargs):
        """Constructor of SARSAAgent

        Initializes a SARSA learning agent.

        :param alpha (float): learning rate for Q-value updates
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def act(self, obs: int) -> int:
        """Epsilon-greedy action selection

        :param obs (int): current state observation
        :return (int): chosen action
        """
        q_values = [self.q_table[(obs, a)] for a in range(self.action_space.n)]
        max_q = max(q_values)
        max_actions = [a for a, q in enumerate(q_values) if q_values[a] == max(q_values)]

        if random.random() < self.epsilon:
            return random.choice(range(self.action_space.n))  # Exploration
        else:
            return random.choice(max_actions)  # Exploitation

    def learn(self, obs: int, action: int, reward: float, n_obs: int, n_action: int, done: bool) -> float:
        """Updates Q-table using SARSA update rule

        :param obs (int): current observation
        :param action (int): current action
        :param reward (float): immediate reward received
        :param n_obs (int): next observation
        :param n_action (int): next action chosen
        :param done (bool): whether the next state is terminal
        :return (float): updated Q-value
        """
        target_value = reward + self.gamma * (1 - done) * self.q_table[(n_obs, n_action)]
        self.q_table[(obs, action)] += self.alpha * (target_value - self.q_table[(obs, action)])

        return self.q_table[(obs, action)]
    
    # uncomment the technique you want to use -> first one is the one provided in the exercise,
    #  second one is the one I implemented on linear decay 

    # this is the given linear decay technique
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates epsilon hyperparameter

        **DO NOT CHANGE THIS FUNCTION FOR PROVIDED TESTS**

        :param timestep (int): current timestep
        :param max_timestep (int): maximum timesteps allowed
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.20 * max_timestep))) * 0.99

    # this is the second technique I implemented on linear decay
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates epsilon hyperparameter based on linear decay to slow down exploration"""
        start_epsilon = 1.0
        end_epsilon = 0.05
        decay_duration = 0.5 * max_timestep  # decay over 50% of training

        decay_ratio = min(1.0, timestep / decay_duration)
        self.epsilon = start_epsilon - decay_ratio * (start_epsilon - end_epsilon)

