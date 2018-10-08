# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize  # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import get_policies


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongDeterministic-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=1, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=True, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=1, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--only_human_state', default=True, type=bool, help='renders the atari environment')
    return parser.parse_args()

discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end)
    f = open(args.save_dir + 'performance_log.txt', mode)
    f.write(s + '\n')
    f.close()


class TDPolicy(nn.Module):  # an actor-critic neural network for third party
    def __init__(self, channels, memsize, num_actions=2):
        super(TDPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5 + 1, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx, theta = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = torch.cat((x.view(-1, 32 * 5 * 5), torch.tensor([[theta]])), dim=1)
        hx = self.gru(x, (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load_td(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class TDPolicy_h(nn.Module):  # a third party policy trained only based on human state
    def __init__(self, num_actions=2):
        super(TDPolicy_h, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 10)
        self.critic_linear, self.actor_linear = nn.Linear(10, 1), nn.Linear(10, num_actions)

    def forward(self, inputs):
        inputs = torch.tensor(inputs)
        inputs = inputs.view(-1, 1)
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        return self.critic_linear(x), self.actor_linear(x)

    def try_load_td(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class NNPolicy(nn.Module):  # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx


def try_load_all(save_dir):
    paths = glob.glob(save_dir + '*.tar')
    step = 0
    if len(paths) > 0:
        index = [int(s.split('.')[-2]) for s in paths]
        pairs = [_ for _ in zip(index, paths)]
        for i in pairs:
            print(i)
        pairs = pairs
        # self.load_state_dict(torch.load(paths[ix]))
    else:
        print("\tno saved models")


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    print("actions", actions)
    print("torch.tensor(actions).view(-1, 1)", torch.tensor(actions).view(-1, 1))
    print('logps', logps)
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    print('after gather', logpys)
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum()  # encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def test(pairs, args):
    env = gym.make(args.env)

    humam_policies = get_policies.get_policies()
    human_paths_rewards = [(i[1], i[2]) for i in humam_policies]
    machine_path_reward = human_paths_rewards[len(human_paths_rewards) // 2]

    highest_reward = 21  # used to normalize theta to [-1, 1]. theta = reward / highest_reward

    human_states_thetas = [(torch.load(path), reward / highest_reward) for path, reward in human_paths_rewards]
    machine_state_theta = (torch.load(machine_path_reward[0]), machine_path_reward[1] / highest_reward)
    num_human_state = len(human_states_thetas)
    human_index = num_human_state - 1  # init_human_index
    theta = human_states_thetas[human_index][1]

    if args.only_human_state:
        td_model = TDPolicy_h(num_actions=2)  # a local model for third party
    else:
        td_model = TDPolicy(channels=1, memsize=args.hidden, num_actions=2)  # a local model for third party
    hm_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)
    for i in range(len(pairs)):
        episodes, epr,  done = 0, 0, False
        state = torch.tensor(prepro(env.reset()))
        index, path = pairs[i]
        td_model.load_state_dict(torch.load(path))
        if not args.only_human_state:
            td_hx = torch.zeros(1, 256) if done else td_hx.detach()  # initialize hidden state for third party
        hm_hx = torch.zeros(1, 256)
        td_values, td_logps, td_actions, td_rewards = [], [], [], []
        f = open(args.save_dir + 'performance_log.txt', 'a')
        while episodes < 200:
            with torch.no_grad():
                if args.only_human_state:
                    td_value, td_logit = td_model(theta)
                else:
                    td_value, td_logit, td_hx = td_model((state.view(1, 1, 80, 80), td_hx, theta))
                td_logp = F.log_softmax(td_logit, dim=-1)
                td_action = torch.exp(td_logp).multinomial(num_samples=1).data[0]
                if td_action.numpy()[0] == 1:
                    hm_model.load_state_dict(human_states_thetas[human_index][0])
                    possibility = [(i + 1) / sum(range(1, human_index + 2)) for i in range(human_index + 1)]
                    human_index = np.random.choice(range(human_index + 1), 1, p=possibility)[0]
                    theta = human_states_thetas[human_index][1]
                else:
                    hm_model.load_state_dict(machine_state_theta[0])
                    possibility = [(num_human_state - i) / sum(range(1, num_human_state - human_index + 1)) for i in
                                   range(human_index, num_human_state)]
                    human_index = np.random.choice(range(human_index, num_human_state), 1, p=possibility)[0]
                    theta = human_states_thetas[human_index][1]

                hm_value, hm_logit, hm_hx = hm_model((state.view(1, 1, 80, 80), hm_hx))
                hm_logp = F.log_softmax(hm_logit, dim=-1)
                hm_action = torch.exp(hm_logp).multinomial(num_samples=1).data[0]
                state, reward, done, _ = env.step(hm_action.numpy()[0])
                if args.render: env.render()
                state = torch.tensor(prepro(state))
                epr += reward
                if done:
                    print(episodes, epr)
                    f.write(str(episodes) + str(epr) + '\n')
                    episodes += 1
                    td_rewards.append(epr)
                    state = torch.tensor(prepro(env.reset()))
                    epr = 0
        rewards_mean = np.mean(td_rewards)
        print('index=%d, path='%index + path + ', rewards_mean=%f' % rewards_mean)
        f.write('index=%d, path='%index + path + ', rewards_mean=%f' % rewards_mean + '\n')
        f.close()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise Exception("Must be using Python 3 with linux!")  # or else you get a deadlock in conv2d

    args = get_args()
    torch.manual_seed(args.seed)
    if args.only_human_state:
        args.save_dir = '{}_td_h/'.format(args.env.lower())  # keep the directory structure simple
        print('\n\tonly use human state as an input for td_policy')
        print('\tdir is saved at', args.save_dir)
        shared_td_model = TDPolicy_h(num_actions=2).share_memory()
    else:
        args.save_dir = '{}_td/'.format(args.env.lower())  # keep the directory structure simple
        print('\n\tuse both human state and physical state as input for td_policy')
        print('\tdir is saved at', args.save_dir)
        shared_td_model = TDPolicy(channels=1, memsize=args.hidden, num_actions=2).share_memory()

    if args.render:  args.processes = 1; args.test = True  # render mode -> test mode w one process
    if args.test:  args.lr = 0  # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n  # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # make dir to save models etc.

    paths = glob.glob(args.save_dir + '*.tar')
    if len(paths) > 0:
        index = [int(s.split('.')[-2]) for s in paths]
        pairs = [_ for _ in zip(index, paths)]
        pairs.sort()
        for i, j in enumerate(pairs):
            print(i, j)
    else:
        raise Exception("No model to test")

    for pair in pairs:
        if pair[0] == 20:
            test([pair], args)
            break

    # test_pairs = pairs[19:]
    # test(test_pairs, args)
