import numpy as np
import torch
from torch.optim import Adam
import argparse
import time

from networks import MetaLearnerNetwork, EmbeddingNetwork, Agent
from utils import ParameterBandit, combined_shape, discount_cumsum


class DataBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma = gamma

    def store(self, obs, act, rew, logp, done):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = self.rew_buf[path_slice]
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):

        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    logp=self.logp_buf, rew=self.rew_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def train_agent(env, meta_net, embed_net, lr, kl_cost, lifetime=1e6):

    # Get environment parameters
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Agent network
    agent = Agent(observation_space=env.observation_space, action_space=env.action_space, m=args.m)

    # Set up optimizer for agent
    agent_optim = Adam(agent.pi.parameters(), lr=lr)

    # Set up data buffer
    buf = DataBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.trajectory_steps, gamma=args.gamma)

    # Update function
    def update():
        # Get data
        data = buf.get()
        obs, act, rew, done = data['obs'], data['act'], data['rew'], data['done']

        # Roll the observation vector to get s_t+1
        obs1 = torch.roll(obs, shifts=-1, dims=1)

        for _ in range(args.train_pi_iters):
            agent_optim.zero_grad()

            # Obtain the y vector for s_t and s_t+1
            pi, logp, y = agent.pi(obs, act)
            _, _, y1 = agent.pi(obs1)

            # Compute the embeddings for each y vector
            fi_y = embed_net(y)
            fi_y1 = embed_net(y1)

            # Compute pi_hat and y_hat
            # TODO: FIX THIS
            inp = [rew, done, args.gamma, torch.exp(logp), fi_y, fi_y1]
            h = None
            c = None
            pi_hat, y_hat = meta_net(inp, h, c)

            # Compute agent loss
            kl_term = torch.mul(y, torch.log(torch.div(y, y_hat)))
            loss = logp * pi_hat - kl_cost * torch.sum(kl_term, dim=-1)
            loss = loss.mean()

            # Optimize
            loss.backward()
            agent_optim.step()

            # TODO: RETURN METAGRADIENTS
            return None

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Metagradients and returns
    metagrads, returns = [], []

    # Main loop: collect experience in env
    for t in range(int(lifetime)):

        # Agent take action
        a, logp = agent.step(torch.as_tensor(o, dtype=torch.float32))

        # Environment next step
        next_o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Save in buffer
        buf.store(o, a, r, logp, d)

        # Update obs
        o = next_o

        # Resetting the episode if current ended
        if d:
            buf.finish_path()
            returns.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if (t + 1) % args.trajectory_steps == 0:
            grad = update()
            metagrads.append(grad)

    return metagrads, returns


def train_lpg(env_dist, init_agent_param_dist, num_meta_iterations=5, num_lifetimes=1, seed=0):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create parameter bandit
    parameter_bandit = ParameterBandit(env_dist)

    # Meta and Embedding networks
    meta_net = MetaLearnerNetwork(inp_dim=6, hidden_size=args.lstm_hidden_size, out_dim=args.m)
    embed_net = EmbeddingNetwork(y_dim=args.m)

    # Set up optimizer for Meta and Embedding
    meta_optim = Adam(meta_net.parameters(), lr=args.meta_lr)
    # TODO: UPDATE RULE FOR EMBED
    embed_optim = Adam(embed_net.parameters(), lr=0.001)

    for _ in range(num_meta_iterations):

        # Initialize meta optim
        meta_optim.zero_grad()

        lifetimes_metagrads = []

        for lifetime in range(num_lifetimes):

            env_name, comb = parameter_bandit.sample_combination()
            lr, kl_cost = comb
            env = env_dist.create(env_name)
            # TODO: LIFETIME VALUE
            metagrads, returns = train_agent(env, meta_net, embed_net, lr, kl_cost)
            lifetimes_metagrads.append(metagrads)

            # Update bandit
            parameter_bandit.update_bandits(env_name, comb, np.mean(returns))

        # TODO: DO THIS CORRECTLY
        loss = torch.sum(metagrads)
        loss.backward()
        meta_optim.step()


def lpg():
    env_dist = None
    init_agent_param_dist = None
    train_lpg(env_dist=env_dist, init_agent_param_dist=init_agent_param_dist,
              num_meta_iterations=args.num_meta_iterations)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main script for running LPG")
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--m', type=int, default=30)
    parser.add_argument('--meta_lr', type=float, default=0.0001)
    parser.add_argument('--train_pi_iters', type=int, default=5)
    parser.add_argument('--trajectory_steps', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--num_meta_iterations', type=int, default=5)
    args = parser.parse_args()

    lpg()
