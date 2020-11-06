import numpy as np
from scipy.signal import lfilter
from scipy.stats import sem, t
import matplotlib.pyplot as plt


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

        # Sample which environment
        env_name = np.random.choice(self.environments)

        # Compute probabilities for that environment
        logits = []
        for (lr, kl_cost) in self.combinations[env_name]:
            comb = (env_name, lr, kl_cost)
            logit = (np.mean(self.returns[comb]) + self.exp_bonus / np.sqrt(self.trials[comb])) / self.temp
            logits.append(logit)
        logits = np.exp(logits)
        logits = logits / np.sum(logits)

        # Sample one combination of learning rate and KL cost based on computed probabilities
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
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def moving_average(a, n=10):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def make_plot(data_y, xlab, ylab, path, func=lambda x: x, confidence=0.95, plot_ci=True):
    f, ax = plt.subplots(**{'figsize': (12, 8)})
    ax.grid(color='#c7c7c7', linestyle='--', linewidth=1)

    if len(data_y[0]) > 1:
        means, uppers, lowers = [], [], []
        for i in range(len(data_y)):
            n, m, std_err = len(data_y[i]), np.mean(data_y[i]), sem(data_y[i])
            h = std_err * t.ppf((1 + confidence) / 2, n - 1)
            means.append(m)
            uppers.append(m + h)
            lowers.append(m - h)
        m = func(means)
        ax.plot(m, color='#539caf', alpha=1)
        if plot_ci:
            ax.fill_between(range(len(m)), y1=func(lowers), y2=func(uppers), color='#539caf', alpha=0.4)
    else:
        data = [d[0] for d in data_y]
        ax.plot(func(data), color='#539caf', alpha=1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(-1, 1)
    f.savefig(path, bbox_inches='tight')
    plt.close(fig=f)
