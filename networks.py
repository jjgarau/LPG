import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def __init__(self, inp_dim, hidden_size, y_dim, round_y=False, device='cpu', num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.round_y = round_y
        self.num_layers = num_layers

        # Meta network
        self.net = nn.GRU(input_size=inp_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc_y = nn.Linear(hidden_size, y_dim)
        self.fc_pi = nn.Linear(hidden_size, 1)

        # Embedding network
        self.embed_fc1 = nn.Linear(y_dim, 16)
        self.embed_fc2 = nn.Linear(16, 1)

        self.device = device

    def embed_y(self, y):
        y = self.embed_fc1(y)
        y = F.relu(y)
        y = self.embed_fc2(y)
        return torch.sigmoid(y)

    def get_estimations(self, inp):
        with torch.no_grad():
            pi, y = self.forward(inp)
        return pi, y

    def forward(self, inp):

        # Get input
        rew, done, gamma, prob, y, y1 = inp

        # Compute the embeddings for each y vector
        fi_y = self.embed_y(y)
        fi_y1 = self.embed_y(y1)

        # Merge parameters
        batch_size, rollout_size = rew.shape[0], rew.shape[1]
        rew = rew.unsqueeze(dim=-1)
        done = done.unsqueeze(dim=-1)
        prob = prob.unsqueeze(dim=-1)
        gamma = gamma * torch.ones_like(prob)
        input = torch.cat((rew, done, gamma, prob, fi_y, fi_y1), dim=-1)

        # We process the input backwards
        input = torch.flip(input, dims=[1])

        def GRU_packed_seq():

            # Get slices of the different episodes
            slices = torch.nonzero(input[:, :, 1].squeeze()).squeeze()

            # Add a 0 in the first position if the is an unfinished episode in the batch
            if slices[0].item() > 0:
                slices = torch.cat([torch.Tensor([0]), slices]).type(torch.int)
            num_ep = slices.shape[0]

            # Compute the episode lengths
            lengths = slices[1:] - slices[:-1]
            lengths = torch.cat([lengths, torch.Tensor([rollout_size - torch.sum(lengths)])]).type(torch.int)
            max_len = torch.max(lengths).item()

            # Create an empty sequence tensor and fill it with episode data
            seq_tensor = torch.zeros((num_ep, max_len, 6)).to(self.device)
            for i in range(lengths.shape[0] - 1):
                seq_tensor[i, :lengths[i].item(), :] = input[:, slices[i].item():slices[i + 1].item(), :]
            seq_tensor[-1, :lengths[-1].item(), :] = input[:, slices[-1].item():, :]

            # Compute the packed input
            packed_input = pack_padded_sequence(seq_tensor, lengths, batch_first=True, enforce_sorted=False)

            # Do the network pass
            packed_output, _ = self.net(packed_input)

            # Compute padded output
            pad_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

            # Retrieve episode data and merge it into a single tensor
            output = torch.zeros((batch_size, rollout_size, self.hidden_size)).to(self.device)
            for i in range(lengths.shape[0] - 1):
                output[:, slices[i].item():slices[i + 1].item(), :] = pad_output[i, :lengths[i].item(), :]
            output[:, slices[-1].item():, :] = pad_output[-1, :lengths[-1].item(), :]

            return output

        def GRU_pass_for():

            # Initialize h vectors
            h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)

            # We initialize the output
            output = torch.zeros((batch_size, rollout_size, self.hidden_size))

            # GRU pass
            for i in range(rollout_size):
                # Reset h if episode is done
                done_filter = 1 - input[:, i:(i+1), 1]
                done_filter = done_filter.unsqueeze(dim=0).repeat((self.num_layers, 1, self.hidden_size))
                h = h * done_filter
                inp = input[:, i:(i + 1), :]
                out, h = self.net(inp, h)
                output[:, i:(i+1), :] = out

            return output

        # Do a GRU pass using the packed sequence method if the batch size is 1
        output = GRU_packed_seq() if batch_size == 1 else GRU_pass_for()

        # Computing y_hat and pi_hat
        y = self.fc_y(output)
        y = torch.sigmoid(y)
        if self.round_y:
            y = torch.round(y)
        pi = self.fc_pi(output)

        # Flip the input back
        pi, y = torch.flip(pi, [1]), torch.flip(y, [1])
        pi, y = pi.squeeze(dim=-1), y.squeeze(dim=-1)

        return pi, y
