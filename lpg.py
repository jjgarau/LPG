import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class ParameterBandit:

    def __init__(self, env_dist, temp=0.1, exp_bonus=0.2):
        self.environments = [env.name for env in env_dist]
        self.combinations = {env.name: [(lr, kl_cost) for lr in env.lrs for kl_cost in env.kl_costs]
                             for env in env_dist}
        self.returns = {(env_name, lr, kl_cost): [] for env_name in self.environments
                        for (lr, kl_cost) in self.combinations[env_name]}
        self.trials = {(env_name, lr, kl_cost): 0 for env_name in self.environments
                       for (lr, kl_cost) in self.combinations[env_name]}
        self.temp = temp
        self.exp_bonus = exp_bonus

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
        self.returns[comb] = [ret] + self.returns[comb][:9]
        self.trials[comb] += 1


def train_lpg(env_dist, init_agent_param_dist, num_parallel_lifetimes=1):

    parameter_bandit = ParameterBandit(env_dist)

    for lifetime in range(num_parallel_lifetimes):
        env_name, (lr, kl_cost) = parameter_bandit.sample_combination()


def lpg():
    env_dist = None
    init_agent_param_dist = None
    train_lpg(env_dist=env_dist, init_agent_param_dist=init_agent_param_dist)


if __name__ == "__main__":
    lpg()
