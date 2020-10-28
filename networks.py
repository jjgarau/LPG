import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete


class AgentNetwork(nn.Module):

    def __init__(self, obs_dim, m, round_y):
        super().__init__()
        # Neural network to compute y vector
        self.fc_y = nn.Linear(obs_dim, m)
        self.round_y = round_y

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def prediction_vector(self, obs):

        # Network pass
        y = self.fc_y(obs)

        # Sigmoid to bound between [0, 1]
        y = torch.sigmoid(y)

        # Round y if predefined
        return torch.round(y) if self.round_y else y

    def forward(self, obs, act=None):

        # Get probability distribution over actions
        pi = self._distribution(obs)

        # Compute logp for provided actions
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None

        return pi, logp_a


class CategoricalAgentNetwork(AgentNetwork):

    def __init__(self, obs_dim, act_dim, m, round_y=False):
        super().__init__(obs_dim=obs_dim, m=m, round_y=round_y)
        self.fc_act = nn.Linear(obs_dim, act_dim)

    def _distribution(self, obs):
        logits = self.fc_act(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianAgentNetwork(AgentNetwork):

    def __init__(self, obs_dim, act_dim, m, round_y=False):
        super().__init__(obs_dim=obs_dim, m=m, round_y=round_y)
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

    def __init__(self, obs_dim, action_space, m):
        super().__init__()

        # Continuous action spaces
        if isinstance(action_space, Box):
            self.pi = GaussianAgentNetwork(obs_dim, action_space.shape[0], m)

        # Discrete action spaces
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalAgentNetwork(obs_dim, action_space.n, m)

    def step(self, obs):
        with torch.no_grad():
            # Get probability distribution and sample action
            pi = self.pi._distribution(obs)
            a = pi.sample()

            # Compute logp of action
            logp_a = self.pi._log_prob_from_distribution(pi, a)

        return a.numpy(), logp_a.numpy()


class MetaLearnerNetwork(nn.Module):

    def __init__(self, inp_dim, hidden_size, y_dim, round_y=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.round_y = round_y
        # TODO: CLARIFY IF THIS IS BIDIRECTIONAL LSTM OR EPISODE RUNNING BACKWARDS

        # Meta network
        self.net = nn.LSTM(input_size=inp_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc_y = nn.Linear(hidden_size * 2, y_dim)
        self.fc_pi = nn.Linear(hidden_size * 2, 1)

        # Embedding network
        self.embed_fc1 = nn.Linear(y_dim, 16)
        self.embed_fc2 = nn.Linear(16, 1)

    def embed_y(self, y):
        y = self.embed_fc1(y)
        y = F.relu(y)
        y = self.embed_fc2(y)
        return torch.sigmoid(y)

    def get_estimations(self, inp, h=None, c=None):
        with torch.no_grad():
            pi, y = self.forward(inp, h, c)
        return pi, y

    def forward(self, inp, h=None, c=None):

        # Get input
        rew, done, gamma, prob, y, y1 = inp

        # Compute the embeddings for each y vector
        fi_y = self.embed_y(y)
        fi_y1 = self.embed_y(y1)

        # Merge parameters
        batch_size = rew.shape[0]
        gamma = torch.Tensor([gamma]).repeat(batch_size).unsqueeze(dim=-1)
        rew = rew.unsqueeze(dim=-1)
        done = done.unsqueeze(dim=-1)
        prob = prob.unsqueeze(dim=-1)
        input = torch.cat((rew, done, gamma, prob, fi_y, fi_y1), dim=-1)
        # TODO: SEQUENCE DIM
        input = input.unsqueeze(dim=1)

        # Initialize h and c vectors
        h = torch.zeros((2, batch_size, self.hidden_size)) if h is None else h
        c = torch.zeros((2, batch_size, self.hidden_size)) if c is None else c

        # Done mask
        mask = torch.roll(done, shifts=1, dims=0)
        mask = mask.repeat((2, 1, self.hidden_size))
        mask = torch.ones_like(mask) - mask
        h = torch.mul(h, mask)
        c = torch.mul(c, mask)

        # LSTM pass
        # TODO: WHAT DO WE DO WITH H AND C? RESET AFTER EACH EPISODE
        out, (h, c) = self.net(input, (h, c))

        # Computing y_hat and pi_hat
        y = self.fc_y(out)
        y = torch.sigmoid(y)
        if self.round_y:
            y = torch.round(y)
        pi = self.fc_pi(out)
        pi, y = pi.squeeze(), y.squeeze()

        return pi, y
