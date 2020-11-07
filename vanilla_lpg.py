#!/usr/bin/python3

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from environments import get_env_dist
from networks import MetaLearnerNetwork, Agent
from utils import ParameterBandit, combined_shape, discount_cumsum, moving_average, make_plot
import os
import datetime
import json
aux = []

class DataBuffer:

    def __init__(self, obs_dim, act_dim, buf_size, gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.gamma = gamma

    def store(self, obs, act, rew, done):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
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
                    rew=self.rew_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def train_agent(env_list, meta_net, lr, kl_cost, lifetime_timesteps=1e3, beta0=0.01, beta1=0.001, beta2=0.001,
                beta3=0.001):

    # Get environment parameters
    obs_dim = len(env_list[0].observation_space.shape) + 1
    act_dim = len(env_list[0].action_space.shape) + 1

    # Agent network
    agent = Agent(obs_dim=obs_dim, action_space=env_list[0].action_space, m=args.m)

    # Set correct number of trajectory steps
    trajectory_steps = args.trajectory_steps + 1
    env_batch_size = len(env_list)

    # Set up data buffer
    buf_list = [DataBuffer(obs_dim=obs_dim, act_dim=act_dim, buf_size=trajectory_steps, gamma=args.gamma)
                for _ in range(env_batch_size)]

    # Function to collect data from buffers
    def collect_data():
        obs, act, rew, done, ret = [], [], [], [], []
        for buf in buf_list:
            data = buf.get()
            obs.append(data['obs'])
            act.append(data['act'])
            rew.append(data['rew'].unsqueeze(dim=-1))
            done.append(data['done'].unsqueeze(dim=-1))
            ret.append(data['ret'].unsqueeze(dim=-1))

        obs = torch.transpose(torch.cat(obs, dim=-1), 0, 1).to(device)
        obs1 = obs[:, 1:]
        act = torch.transpose(torch.cat(act, dim=-1), 0, 1)[:, :-1].to(device)
        rew = torch.transpose(torch.cat(rew, dim=-1), 0, 1)[:, :-1].to(device)
        done = torch.transpose(torch.cat(done, dim=-1), 0, 1)[:, :-1].to(device)
        ret = torch.transpose(torch.cat(ret, dim=-1), 0, 1)[:, :-1].to(device)
        obs = obs[:, :-1]

        return obs.unsqueeze(dim=-1), obs1.unsqueeze(dim=-1), act, rew, done, ret

    # Agent update function
    def update_agent(vanilla=False):

        # Get data
        obs, obs1, act, rew, done, ret = collect_data()

        for _ in range(args.train_pi_iters):

            # Clear gradients
            agent.pi.zero_grad()

            # Obtain logp vector for s_t
            _, logp = agent.pi(obs, act.squeeze())

            # Obtain the y vector for s_t and s_t+1
            y = agent.pi.prediction_vector(obs)
            y1 = agent.pi.prediction_vector(obs1)
            y, y1 = torch.zeros_like(y), torch.zeros_like(y1)

            # Compute pi_hat and y_hat
            inp = (rew, done, args.gamma, torch.exp(logp), y, y1)
            pi_hat, y_hat = meta_net.get_estimations(inp)

            # Compute agent loss
            kl_term = torch.sum(F.kl_div(y, y_hat, reduction='none'), dim=-1)
            if vanilla:
                loss = logp * ret
            else:
                loss = logp * pi_hat  # - 0.001 * kl_cost * kl_term
            loss = loss.mean()

            # Optimize
            g = torch.autograd.grad(loss, agent.pi.parameters(), retain_graph=True, allow_unused=True)
            state_dict = agent.pi.state_dict()
            for i, (name, param) in enumerate(state_dict.items()):
                # Gradient ascent
                if g[i] is not None:
                    state_dict[name] = state_dict[name] + lr * g[i]
            agent.pi.load_state_dict(state_dict)

    def get_meta_gradient(eps=1e-7, eps_ent=1e-3, can_break=True):

        # Get data
        obs, obs1, act, rew, done, ret = collect_data()

        # Obtain logp vector for s_t
        pi, logp = agent.pi(obs, act.squeeze())

        # Compute pi entropy
        ent_pi = pi.entropy()

        # Break if policy becomes deterministic
        if ent_pi.mean().item() < eps_ent and can_break:
            return None

        # Obtain the y vector for s_t and s_t+1
        y = agent.pi.prediction_vector(obs)
        y1 = agent.pi.prediction_vector(obs1)
        y, y1 = torch.zeros_like(y), torch.zeros_like(y1)

        # Compute y entropy
        ent_y = - y * torch.log2(y + eps) - (1 - y) * torch.log2(1 - y + eps)
        ent_y = torch.mean(ent_y, dim=-1)

        # Break if policy becomes deterministic
        if ent_y.mean().item() < eps_ent and can_break:
            return None

        # Compute pi_hat and y_hat
        inp = (rew, done, args.gamma, torch.exp(logp), y, y1)
        pi_hat, y_hat = meta_net(inp)

        # Compute L2 norms
        l2_pi = torch.pow(pi_hat, 2)
        l2_y = torch.sum(torch.pow(y_hat, 2), dim=-1)

        # Compute meta gradients
        # meta_grad = logp * ret + beta0 * ent_pi + beta1 * ent_y - beta2 * l2_pi - beta3 * l2_y
        meta_grad = -1 * F.mse_loss(ret, pi_hat.unsqueeze(dim=0), reduction='none')
        meta_grad = meta_grad.mean()

        aux.append(-1 * meta_grad.item())

        meta_grad = torch.autograd.grad(meta_grad, meta_net.parameters(), retain_graph=False, allow_unused=True)

        # return meta_grad.mean()
        return meta_grad

    # Prepare for interaction with environment
    ep_ret = [0 for _ in range(env_batch_size)]
    ep_len = [0 for _ in range(env_batch_size)]
    o = np.array([env.reset() for env in env_list])

    # Metagradients and returns
    single_env_returns = [[] for _ in range(env_batch_size)]
    meta_gradients = None
    meta_counter = 0
    agent_turn = True

    # Main loop: collect experience in env
    for t in range(int(lifetime_timesteps)):

        # Agent take action
        a, logp = agent.step(torch.as_tensor(o, dtype=torch.float32))

        # Environment next step
        for i, env in enumerate(env_list):

            next_o, r, d, _ = env.step(a[i])
            ep_ret[i] += r
            ep_len[i] += 1

            # Save in buffer
            buf_list[i].store(o[i], a[i], r, d)

            # Update obs
            o[i] = next_o

            # Resetting the episode if current ended
            if d:
                buf_list[i].finish_path()
                single_env_returns[i].append(ep_ret[i])
                o[i], ep_ret[i], ep_len[i] = env.reset(), 0, 0

        # Print iteration number
        if (t + 1) % 10000 == 0:
            print(t + 1)

        # Call update function after K steps
        if (t + 1) % trajectory_steps == 0:

            agent.to(device)
            agent.pi.to(device)

            if agent_turn:

                # Do agent update based on rollout
                update_agent(vanilla=args.vanilla)
                agent_turn = False

            else:

                # Compute meta gradient of the rollout
                can_break = True if t + 1 > 10 * trajectory_steps and not args.vanilla else False
                can_break = False
                meta_grad = get_meta_gradient(can_break=can_break)

                # Break to prevent early divergence
                if meta_grad is None:
                    break

                # Sum meta gradients
                if meta_gradients is None:
                    meta_gradients = list(meta_grad)
                else:
                    for i in range(len(meta_grad)):
                        if meta_grad[i] is not None:
                            meta_gradients[i] = meta_gradients[i] + meta_grad[i]

                meta_counter += 1
                agent_turn = True

            agent.to('cpu')
            agent.pi.to('cpu')

    for i in range(len(meta_gradients)):
        if meta_gradients[i] is not None:
            meta_gradients[i] = meta_gradients[i] / meta_counter

    # return meta_gradients / meta_counter, np.mean(returns)
    return meta_gradients, single_env_returns


def train_lpg(env_dist, init_agent_param_dist, num_meta_iterations=5, num_lifetimes=1, seed=0, results_folder=None):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create parameter bandit
    parameter_bandit = ParameterBandit(env_dist)
    env_dict = {env.name: env for env in env_dist}

    # TODO: CODE THE INITIAL PARAMETER DISTRIBUTION, distribution on theta

    # Meta and Embedding networks
    meta_net = MetaLearnerNetwork(inp_dim=6, hidden_size=args.lstm_hidden_size, y_dim=args.m, device=device)
    meta_net.to(device)

    # Set up optimizer for Metanetwork
    # meta_optim = Adam(meta_net.parameters(), lr=args.meta_lr)

    # Tracking returns
    all_returns = []

    for it in range(num_meta_iterations):

        print("Meta iteration", it + 1)

        # Initialize meta optim
        # meta_optim.zero_grad()

        # Clear gradients
        meta_net.zero_grad()

        lifetimes_meta_losses, lifetimes_returns = [], []

        for lifetime in range(num_lifetimes):

            print("Starting lifetime", lifetime + 1)

            # Sample an environment, a learning rate, and a kl cost value
            env_name, comb = parameter_bandit.sample_combination()
            lr, kl_cost = comb

            # Create the environment
            env_list = [env_dict[env_name]() for _ in range(args.parallel_environments)]

            # Train the agent and receive the metagradients
            meta_losses, single_env_returns = train_agent(env_list, meta_net, lr, kl_cost,
                                                          lifetime_timesteps=args.lifetime_timesteps, beta0=args.beta0,
                                                          beta1=args.beta1, beta2=args.beta2, beta3=args.beta3)
            lifetimes_meta_losses.append(meta_losses)

            # Compute average return per environment and per episode
            returns_env = [np.mean(ret) for ret in single_env_returns]
            returns = np.mean(returns_env)
            lifetimes_returns = returns_env

            # Update bandit
            parameter_bandit.update_bandits(env_name, comb, returns)

        all_returns.append(lifetimes_returns)

        # Gradient ascent
        # loss = -1 * sum(lifetimes_meta_losses) / len(lifetimes_meta_losses)
        # loss.backward()
        # meta_optim.step()

        import matplotlib.pyplot as plt
        plt.plot(aux)
        plt.savefig(os.path.join(results_folder, 'loss.pdf'))
        plt.close()
        # aux.clear()

        state_dict = meta_net.state_dict()
        for i, (name, param) in enumerate(state_dict.items()):
            # Gradient ascent
            g = sum(m[i] for m in lifetimes_meta_losses if m[i] is not None)
            if g is not None:
                state_dict[name] = state_dict[name] + args.meta_lr * g
        meta_net.load_state_dict(state_dict)

        make_plot(data_y=all_returns, xlab='Meta iteration', ylab='Average return over lifetime',
                  path=os.path.join(results_folder, 'returns.pdf'), plot_ci=False)

        if len(all_returns) > 10:
            make_plot(data_y=all_returns, xlab='Meta iteration',
                      ylab='Moving average of return over lifetime',
                      path=os.path.join(results_folder, 'returns_ma.pdf'), func=lambda x: moving_average(x, n=10),
                      plot_ci=False)

        torch.save(meta_net, os.path.join(results_folder, 'model.pkl'))


def lpg():
    os.makedirs('results', exist_ok=True)
    results_folder = os.path.join('results', 'simulation_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    os.makedirs(results_folder, exist_ok=True)

    sim_params = vars(args)
    with open(os.path.join(results_folder, 'arguments.txt'), 'w') as file:
        file.write(json.dumps(sim_params))

    env_dist = get_env_dist()
    init_agent_param_dist = None
    train_lpg(env_dist=env_dist, init_agent_param_dist=init_agent_param_dist,
              num_meta_iterations=args.num_meta_iterations, num_lifetimes=args.num_lifetimes,
              results_folder=results_folder)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description="Main script for running LPG")
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help="Hidden size of the LSTM meta network")
    parser.add_argument('--m', type=int, default=5, help="Dimension of y vector")
    parser.add_argument('--meta_lr', type=float, default=0.1, help="Learning rate for the meta network")
    parser.add_argument('--train_pi_iters', type=int, default=5,
                        help="K, number of consecutive training iterations for the agent")
    parser.add_argument('--trajectory_steps', type=int, default=20, help="Number of steps between agent iterations")
    parser.add_argument('--gamma', type=float, default=0.995, help="Discount factor")
    parser.add_argument('--num_meta_iterations', type=int, default=5000, help="Number of meta updates")
    parser.add_argument('--num_lifetimes', type=int, default=1, help="Number of parallel lifetimes")
    parser.add_argument('--lifetime_timesteps', type=int, default=3e3, help="Number of timesteps per lifetime")
    parser.add_argument('--parallel_environments', type=int, default=1, help="Number of parallel environments")
    parser.add_argument('--beta0', type=float, default=0.01, help="Policy entropy cost, beta 0")
    parser.add_argument('--beta1', type=float, default=0.001, help="Prediction entropy cost, beta 1")
    parser.add_argument('--beta2', type=float, default=0.001, help="L2 regularization weight for pi hat, beta 2")
    parser.add_argument('--beta3', type=float, default=0.001, help="L2 regularization wright for y hat, beta 3")
    parser.add_argument('--vanilla', type=bool, default=False, help="Run a vanilla LPG to debug")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('CUDA Available:', torch.cuda.is_available())

    lpg()
