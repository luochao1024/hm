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
os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongDeterministic-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=16, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--only_human_state', default=True, type=bool, help='only use human state to train td policy')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end)
    f = open(args.save_dir + 'log.txt', mode)
    f.write(s + '\n')
    f.close()


class NNPolicy(nn.Module):  # an actor-critic neural network for machine
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

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


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
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
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


class SharedAdam(torch.optim.Adam):  # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1  # a "step += 1"  comes later
            super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum()  # encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def train(shared_td_model, shared_optimizer, human_states_thetas,
          machine_state_theta, rank, args, info, init_human_index):
    env = gym.make(args.env)  # make a local (unshared) environment
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)  # seed everything
    if args.only_human_state:
        td_model = TDPolicy_h(num_actions=2)  # a local model for third party
    else:
        td_model = TDPolicy(channels=1, memsize=args.hidden, num_actions=2)  # a local model for third party
    hm_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)  # a model for human or machine
    state = torch.tensor(prepro(env.reset()))  # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True  # bookkeeping

    num_human_state = len(human_states_thetas)
    human_index = init_human_index
    theta = human_states_thetas[human_index][1]

    while info['iterations'][0] <= 8e0 or args.test:
        td_model.load_state_dict(shared_td_model.state_dict())  # sync with shared model

        if not args.only_human_state:
            td_hx = torch.zeros(1, 256) if done else td_hx.detach()  # initialize hidden state for third party

        hm_hx = torch.zeros(1, 256)  # initialize hidden state for hm_model
        values, logps, actions, rewards = [], [], [], []  # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1
            if args.only_human_state:
                value, logit = td_model(theta)
            else:
                value, logit, td_hx = td_model((state.view(1, 1, 80, 80), td_hx, theta))

            logp = F.log_softmax(logit, dim=-1)
            action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
            if action.numpy()[0] == 1:
                hm_model.load_state_dict(human_states_thetas[human_index][0])
                # possibility = [i**2 / sum(np.square(range(1, human_index + 2))) for i in range(1, human_index + 2)]
                possibility = [i**3 / sum(np.power(range(1, human_index + 2), 3)) for i in range(1, human_index + 2)]
                human_index = np.random.choice(range(human_index+1), 1, p=possibility)[0]
                theta = human_states_thetas[human_index][1]
            else:
                hm_model.load_state_dict(machine_state_theta[0])
                # possibility =[(num_human_state - i)**2 / sum(np.square(range(1, num_human_state - human_index + 1))) for i in range(human_index, num_human_state)]
                possibility =[(num_human_state - i)**3 / sum(np.power(range(1, num_human_state - human_index + 1), 3)) for i in range(human_index, num_human_state)]
                human_index = np.random.choice(range(human_index, num_human_state), 1, p=possibility)[0]
                theta = human_states_thetas[human_index][1]

            reward = 0  # reset reward to 0
            while reward == 0:
                hm_value, hm_logit, hm_hx = hm_model((state.view(1, 1, 80, 80), hm_hx))
                hm_logp = F.log_softmax(hm_logit, dim=-1)
                hm_action = torch.exp(hm_logp).multinomial(num_samples=1).data[0]
                state, reward, done, _ = env.step(hm_action.numpy()[0])
                state = torch.tensor(prepro(state))
                info['frames'].add_(1)

            info['iterations'].add_(1)
            if args.render: env.render()

            epr += reward
            reward = np.clip(reward, -1, 1)  # reward
            done = done or episode_length >= 1e4  # don't playing one ep for too long

            # num_frames = int(info['frames'].item())
            num_iterations = int(info['iterations'].item())
            if num_iterations % 5e4 == 0:  # save every 2M frames
                printlog(args, '\n\t{:.0f}F frames: saved td_model\n'.format(num_iterations / 1e4))
                torch.save(shared_td_model.state_dict(), args.save_dir + 'td_model.{:.0f}.tar'.format(num_iterations / 1e4))

            if done:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args,
                         'time {}, episodes {:.0f}, frames {:.1f}*10^5, exp running mean of reward {:.2f}, run loss {:.2f}, num_iterations {:.1f}*10^3'
                         .format(elapsed, info['episodes'].item(), int(info['frames'].item()) / 1e5,
                                 info['run_epr'].item(), info['run_loss'].item(), num_iterations/1e3))
                last_disp_time = time.time()

            if done:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        if args.only_human_state:
            next_value = torch.zeros(1, 1) if done else td_model(theta)[0]
        else:
            next_value = torch.zeros(1, 1) if done else td_model((state.unsqueeze(0), td_hx, theta))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(td_model.parameters(), 40)

        for param, shared_param in zip(td_model.parameters(), shared_td_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise Exception("Must be using Python 3 with linux!")  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = '{}_td_h/'.format(args.env.lower())  # keep the directory structure simple
    if args.render:  args.processes = 1; args.test = True  # render mode -> test mode w one process
    if args.test:  args.lr = 0  # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n  # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # make dir to save models etc.

    torch.manual_seed(args.seed)
    if args.only_human_state:
        print('\n\tonly use human state as an input for td_policy\n')
        print('\n\tdir is saved at\n', args.save_dir)
        shared_td_model = TDPolicy_h(num_actions=2).share_memory()
    else:
        print('\n\tuse both human state and phisical state as input for td_policy\n')
        print('\n\tdir is saved at\n', args.save_dir)
        shared_td_model = TDPolicy(channels=1, memsize=args.hidden, num_actions=2).share_memory()

    shared_optimizer = SharedAdam(shared_td_model.parameters(), lr=args.lr)

    humam_policies = get_policies.get_policies()
    human_paths_rewards = [(i[1], i[2]) for i in humam_policies]
    machine_path_reward = human_paths_rewards[len(human_paths_rewards)//2]
    init_human_index = len(human_paths_rewards)-19
    highest_reward = 21  # used to normalize theta to [-1, 1]. theta = reward / highest_reward

    human_states_thetas = [(torch.load(path), reward/highest_reward) for path, reward in human_paths_rewards]
    machine_state_theta = (torch.load(machine_path_reward[0]), machine_path_reward[1]/highest_reward)
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames', 'iterations']}
    info['iterations'] += shared_td_model.try_load_td(args.save_dir) * 1e4
    if int(info['iterations'].item()) == 0: printlog(args, '', end='', mode='w')  # clear log file

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_td_model, shared_optimizer, human_states_thetas,
                                           machine_state_theta, rank, args, info, init_human_index))
        p.start()
        processes.append(p)
    for p in processes: p.join()
