from gym import Env
import random
from gym.spaces import Discrete
import numpy as np


lower = 5
upper_short = 10
upper_long = 15


class DelayedChainMDP(Env):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def lowerbound(self):
        raise NotImplementedError

    @property
    def upperbound(self):
        raise NotImplementedError

    @property
    def lrs(self):
        raise NotImplementedError

    @property
    def kl_costs(self):
        raise NotImplementedError

    def noisy_reward(self):
        return 0

    def __init__(self):
        self.counter = 0  # the episode ends after delayed_chain_length steps
        self.initial_action = -1
        self.initial_target_state = random.randint(0, 1)  # the first decision determines the reward
        self.chain_length = random.randint(self.lowerbound, self.upperbound)
        self.action_space = Discrete(2)
        self.observation_space = Discrete(self.chain_length)

    def _get_observation(self):
        obs = np.zeros(shape=1, dtype=np.float32)
        if self.counter == 0:
            obs[0] = 0
        else:
            obs[0] = 2 * self.counter - self.initial_action  # states on the first chain are even numbered, others are odd-numbered
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
        if self.counter <= 1:
            self.initial_action = action
        elif self.counter == self.chain_length:
            done = True
            reward = 1 if self.initial_action == self.initial_target_state else -1

        if reward == 0:
            reward += self.noisy_reward()

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
        self.initial_action = -1
        # self.initial_target_state = random.randint(0, 1)  # the first decision determines the reward

        return self._get_observation()


class ShortDelayedChainMDP(DelayedChainMDP):

    name = "short"
    lowerbound = lower
    upperbound = upper_short
    lrs = [20, 40, 80, 160, 320]
    kl_costs = [0.1, 0.5, 1]

class ShortNoisyDelayedChainMDP(ShortDelayedChainMDP):

    name = "short-noisy"

    def noisy_reward(self):
        return random.choice([-1, 1])


class LongDelayedChainMDP(DelayedChainMDP):

    name = "long"
    lowerbound = lower
    upperbound = upper_long
    lrs = [20, 40, 80]
    kl_costs = [0.1, 0.5, 1]

class LongNoisyDelayedChainMDP(LongDelayedChainMDP):

    name = "long-noisy"

    def noisy_reward(self):
        return random.choice([-1, 1])


# ENVIRONMENTS = [ShortDelayedChainMDP, LongDelayedChainMDP, ShortNoisyDelayedChainMDP, LongNoisyDelayedChainMDP]
ENVIRONMENTS = [ShortDelayedChainMDP]


def get_env_dist():
    return ENVIRONMENTS
