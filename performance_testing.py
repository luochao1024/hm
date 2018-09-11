# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize  # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongDeterministic-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=1, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=1, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end)
    f = open(args.save_dir + 'performance_log.txt', mode)
    f.write(s + '\n')
    f.close()


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
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)
    for i in range(len(pairs)):
        episodes, epr,  done = 0, 0, False
        state = torch.tensor(prepro(env.reset()))
        index, path = pairs[i]
        model.load_state_dict(torch.load(path))
        hx = torch.zeros(1, 256)
        values, logps, actions, rewards = [], [], [], []
        f = open(args.save_dir + 'performance_log.txt', 'a')
        while episodes < 200:
            with torch.no_grad():
                value, logit, hx = model((state.view(1, 1, 80, 80), hx))
                logp = F.log_softmax(logit, dim=-1)
                action = torch.exp(logp).multinomial(num_samples=1).data[0]
                state, reward, done, _ = env.step(action.numpy()[0])
                state = torch.tensor(prepro(state))
                epr += reward
                if done:
                    print(episodes, epr)
                    f.write(str(episodes) + str(epr) + '\n')
                    episodes += 1
                    rewards.append(epr)
                    state = torch.tensor(prepro(env.reset()))
                    epr = 0
        rewards_mean = np.mean(rewards)
        print('index=%d, path='%index + path + ', rewards_mean=%f' % rewards_mean)
        f.write('index=%d, path='%index + path + ', rewards_mean=%f' % rewards_mean + '\n')
        f.close()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise Exception("Must be using Python 3 with linux!")  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower())  # keep the directory structure simple
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
    pairs = pairs[19:]
    test(pairs, args)
