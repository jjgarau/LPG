import numpy as np
from scipy.signal import lfilter


class ParameterBandit:

    def __init__(self, env_dist, temp=0.1, exp_bonus=0.2, ret_size=10):
        self.environments = [env.name for env in env_dist]
        self.combinations = {env.name: [(lr, kl_cost) for lr in env.lrs for kl_cost in env.kl_costs]
                             for env in env_dist}
        self.returns = {(env_name, lr, kl_cost): [0] for env_name in self.environments
                        for (lr, kl_cost) in self.combinations[env_name]}
        self.trials = {(env_name, lr, kl_cost): 1 for env_name in self.environments
                       for (lr, kl_cost) in self.combinations[env_name]}
        self.temp = temp
        self.exp_bonus = exp_bonus
        self.ret_size = ret_size

    def sample_combination(self):
        env_name = np.random.choice(self.environments)
        logits = []
        for (lr, kl_cost) in self.combinations[env_name]:
            comb = (env_name, lr, kl_cost)
            logit = (np.mean(self.returns[comb]) + self.exp_bonus / np.sqrt(self.trials[comb])) / self.temp
            logits.append(logit)
        logits = np.exp(logits)
        logits = logits / np.sum(logits)
        index = np.random.choice(len(self.combinations[env_name]), p=logits)
        return env_name, self.combinations[env_name][index]

    def update_bandits_list(self, env_name_list, env_comb_list, ret_list):
        for env_name, env_comb, ret in zip(env_name_list, env_comb_list, ret_list):
            self.update_bandits(env_name, env_comb, ret)

    def update_bandits(self, env_name, env_comb, ret):
        comb = (env_name, *env_comb)
        self.returns[comb] = [ret] + self.returns[comb][:(self.ret_size - 1)]
        self.trials[comb] += 1


def combined_shape(length, shape=None):
    if shape is None:
        return length
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
