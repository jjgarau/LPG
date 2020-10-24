from gym import Env
import random
from gym.spaces import Discrete
import numpy as np


# TODO: make one class that handles all cases
class ShortDelayedChainMDP(Env):

    chain_length = 0
    name = "short"

    lrs = [20.0, 40.0, 80.0]
    kl_costs = [0.1, 0.5, 1.0]

    @classmethod
    def reset_parameters(cls):
        cls.delayed_chain_length = random.randint(5, 30)

    def __init__(self):
        self.counter = 0  # the episode ends after delayed_chain_length steps
        self.initial_action = 0
        self.initial_target_state = random.randint(0, 1)  # the first decision determines the reward
        self.action_space = Discrete(2)
        self.observation_space = Discrete(self.chain_length)

    def _get_observation(self):
        obs = np.zeros(shape=1, dtype=np.float32)
        obs[
            0] = 2 * self.counter + self.initial_action  # states on the first chain are even numbered, others are odd-numbered
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.counter += 1
        done = False
        reward = 0
        if self.counter == 1:
            self.initial_action = action
        elif self.counter == self.chain_length:
            done = True
            reward = 1 if self.initial_action == self.initial_target_state else -1

        obs = self._get_observation()

        return obs, reward, done, None

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        self.counter = 0  # the episode ends after delayed_chain_length steps
        self.initial_action = 0
        self.initial_target_state = random.randint(0, 1)  # the first decision determines the reward

        return self._get_observation()


ENVIRONMENTS = [ShortDelayedChainMDP]


def get_env_dist():
    return ENVIRONMENTS
