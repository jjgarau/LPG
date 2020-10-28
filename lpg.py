import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import argparse
import time
from environments import get_env_dist
from networks import MetaLearnerNetwork, Agent
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

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):

        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    logp=self.logp_buf, rew=self.rew_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def train_agent(env, meta_net, lr, kl_cost, lifetime=1e3):

    # Get environment parameters
    obs_dim = len(env.observation_space.shape) + 1
    act_dim = len(env.action_space.shape) + 1

    # Agent network
    agent = Agent(obs_dim=obs_dim, action_space=env.action_space, m=args.m)

    # Set up data buffer
    buf = DataBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.trajectory_steps, gamma=args.gamma)

    # Agent update function
    def update_agent():
        # Get data
        data = buf.get()
        obs, act, rew, done, ret = data['obs'], data['act'], data['rew'], data['done'], data['ret']

        # Roll the observation vector to get s_t+1
        obs1 = torch.roll(obs, shifts=-1, dims=0)

        meta_loss = 0

        for _ in range(args.train_pi_iters):

            # Obtain logp vector for s_t
            _, logp = agent.pi(obs, act.squeeze())

            # Obtain the y vector for s_t and s_t+1
            merge_obs = torch.cat((obs, obs1), dim=-1).unsqueeze(dim=-1)
            merge_y = agent.pi.prediction_vector(merge_obs)
            y = merge_y[:, 0, :].squeeze()
            y1 = merge_y[:, 1, :].squeeze()

            # Compute pi_hat and y_hat
            inp = (rew, done, args.gamma, torch.exp(logp), y, y1)
            pi_hat, y_hat = meta_net.get_estimations(inp)

            # Compute agent loss
            kl_term = torch.sum(F.kl_div(y, y_hat, reduction='none'), dim=-1)
            loss = logp * pi_hat - kl_cost * kl_term
            loss = loss.mean()

            # Optimize
            g = torch.autograd.grad(loss, agent.pi.parameters(), retain_graph=True)
            state_dict = agent.state_dict()
            for i, (name, param) in enumerate(state_dict.items()):
                # Gradient ascent
                state_dict[name] = state_dict[name] + lr * g[i]

            # TODO: FULL METALOSS FUNCTION
            logp_ret = logp * ret
            meta_loss = meta_loss + logp_ret.mean()

        return meta_loss

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Metagradients and returns
    returns = []
    meta_losses = 0
    meta_counter = 0

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

        # Print iteration number
        if (t + 1) % 100 == 0:
            print(t + 1)

        # Call update function after K steps
        if (t + 1) % args.trajectory_steps == 0:
            meta_loss = update_agent()
            meta_losses = meta_losses + meta_loss
            meta_counter += 1

    return meta_losses / meta_counter, returns


def train_lpg(env_dist, init_agent_param_dist, num_meta_iterations=5, num_lifetimes=1, seed=0):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create parameter bandit
    parameter_bandit = ParameterBandit(env_dist)
    env_dict = {env.name: env for env in env_dist}

    # TODO: CODE THE INITIAL PARAMETER DISTRIBUTION, distribution on theta

    # Meta and Embedding networks
    meta_net = MetaLearnerNetwork(inp_dim=6, hidden_size=args.lstm_hidden_size, y_dim=args.m)

    # Set up optimizer for Metanetwork
    meta_optim = Adam(meta_net.parameters(), lr=args.meta_lr)

    for _ in range(num_meta_iterations):

        # Initialize meta optim
        meta_optim.zero_grad()

        lifetimes_meta_losses = []

        for lifetime in range(num_lifetimes):

            env_name, comb = parameter_bandit.sample_combination()
            lr, kl_cost = comb
            env = env_dict[env_name]()
            meta_losses, returns = train_agent(env, meta_net, lr, kl_cost)
            lifetimes_meta_losses.append(meta_losses)

            # Update bandit
            parameter_bandit.update_bandits(env_name, comb, np.mean(returns))

        loss = -1 * sum(lifetimes_meta_losses) / len(lifetimes_meta_losses)
        loss.backward()
        meta_optim.step()


def lpg():
    env_dist = get_env_dist()
    init_agent_param_dist = None
    train_lpg(env_dist=env_dist, init_agent_param_dist=init_agent_param_dist,
              num_meta_iterations=args.num_meta_iterations, num_lifetimes=args.num_lifetimes)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description="Main script for running LPG")
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help="Hidden size of the LSTM meta network")
    parser.add_argument('--m', type=int, default=30, help="Dimension of y vector")
    parser.add_argument('--meta_lr', type=float, default=0.0001, help="Learning rate for the meta network")
    parser.add_argument('--train_pi_iters', type=int, default=5,
                        help="K, number of consecutive training iterations for the agent")
    parser.add_argument('--trajectory_steps', type=int, default=20, help="Number of steps between agent iterations")
    parser.add_argument('--gamma', type=float, default=0.995, help="Discount factor")
    parser.add_argument('--num_meta_iterations', type=int, default=5, help="Number of meta updates")
    parser.add_argument('--num_lifetimes', type=int, default=1, help="Number of parallel lifetimes")
    args = parser.parse_args()

    lpg()
