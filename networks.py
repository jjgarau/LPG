import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete


class EmbeddingNetwork(nn.Module):

    def __init__(self, y_dim):
        super().__init__()
        self.fc1 = nn.Linear(y_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, y):
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        return torch.sigmoid(y)


class AgentNetwork(nn.Module):

    def __init__(self, obs_dim, m):
        super().__init__()
        self.fc_y = nn.Linear(obs_dim, m)

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def _prediction_vector(self, obs):
        y = self.fc_y(obs)
        y = torch.sigmoid(y)
        return torch.round(y)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        y = self._prediction_vector(obs)
        return pi, logp_a, y


class CategoricalAgentNetwork(AgentNetwork):

    def __init__(self, obs_dim, act_dim, m):
        super().__init__(obs_dim=obs_dim, m=m)
        self.fc_act = nn.Linear(obs_dim, act_dim)
        self.fc_y = nn.Linear(obs_dim, m)

    def _distribution(self, obs):
        logits = self.fc_act(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianAgentNetwork(AgentNetwork):

    def __init__(self, obs_dim, act_dim, m):
        super().__init__(obs_dim=obs_dim, m=m)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Linear(obs_dim, act_dim)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class Agent(nn.Module):

    def __init__(self, observation_space, action_space, m):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = GaussianAgentNetwork(obs_dim, action_space.shape[0], m)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalAgentNetwork(obs_dim, action_space.n, m)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.numpy(), logp_a.numpy()


class MetaLearnerNetwork(nn.Module):

    def __init__(self, inp_dim, hidden_size, out_dim):
        super().__init__()
        self.net = nn.LSTM(input_size=inp_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc_y = nn.Linear(hidden_size * 2, out_dim)
        self.fc_pi = nn.Linear(hidden_size * 2, 1)

    def forward(self, inp, h, c):
        out, (h, c) = self.net(inp, (h, c))
        y = self.fc_y(out)
        y = torch.sigmoid(y)
        y = torch.round(y)
        pi = self.fc_pi(out)
        return out, h, c, y, pi
