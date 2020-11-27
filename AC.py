import  torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as opt
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import  os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v0").unwrapped
np.random.seed(2)
torch.random.seed()
env = env.unwrapped
n_actions = env.action_space.n
n_features = env.observation_space.shape[0]
action_space = np.arange(env.action_space.n)
env.seed(1)
class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(in_features=n_features,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions),
        )
    def forward(self,x):
        x=self.fc(x)
        x=F.softmax(x,dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self,):
       super(Critic, self).__init__()
       self.fc = nn.Sequential(
           nn.Linear(in_features=n_features, out_features=32),
           nn.ReLU(),
           nn.Linear(in_features=32, out_features=128),
           nn.ReLU(),
           nn.Linear(in_features=128, out_features=64),
           nn.ReLU(),
           nn.Linear(in_features=64, out_features=1),
       )

    def forward(self, x):
        x = self.fc(x)
        return x

lr_A=0.000025
lr_C=0.00025
gamma=0.9
class A_C(object):

    def __init__(self):
        super(A_C, self).__init__()
        self.lr_A=lr_A
        self.lr_C=lr_C
        self.dlr=0.00001

        self.actor_net=Actor()
        self.optimizer_A=opt.Adam(self.actor_net.parameters(),lr=self.lr_A)

        self.critic_net=Critic()
        self.optimizer_C=opt.Adam(self.critic_net.parameters(),lr=self.lr_C)


    def choose_actions_of_A(self,s):

        # action_prob=self.actor_net(torch.FloatTensor(s)).detach().numpy()
        # # print(action_prob)
        # action=np.random.choice(action_space,p=action_prob)

        probs = self.actor_net(torch.FloatTensor(s))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        # prob = probs[:, action].view(1, -1)
        # log_prob = prob.log()
        entropy = - (probs * probs.log()).sum()

        return action.numpy(),log_prob, entropy

    # def learning_of_A(self,s,a,td_error,log_prob,entropy):
    #
    #     # log_probs=torch.log(self.actor_net(torch.FloatTensor(s))[a]) ##？？
    #     # print('log_probs',log_probs)
    #     # -log_prob * critic_delta - 0.01 * entropies
    #     loss2 = -td_error.detach()* log_prob- 0.01 * entropy
    #     # print('selected_log_probs',selected_log_probs)
    #     # Loss is negative of expected policy function J = R * log_prob
    #     # loss2 = -selected_log_probs.mean()
    #
    #     self.optimizer_A.zero_grad()
    #     loss2.backward()
    #     self.optimizer_A.step()

    def adjust_lr(self):
        pass
        # for param_group in self.optimizer_A.param_groups:
        #     param_group['lr'] -= self.dlr
        #     if param_group['lr'] <=0.0003:
        #         param_group['lr'] = 0.0003
        #
        # for param_group in self.optimizer_C.param_groups:
        #     param_group['lr'] -= self.dlr
        #     if param_group['lr'] <=0.003:
        #         param_group['lr'] = 0.003

    def learning_of_C(self,s,r,s_,done,log_prob,entropy):

        s,s_=torch.FloatTensor(s),torch.FloatTensor(s_)
        value_s_=self.critic_net(s_)
        value_s=self.critic_net(s)

        td_error= r+value_s_*(1-int(done))*gamma-value_s

        loss_A=-log_prob*td_error.detach()
               # - 0.01*entropy

        loss_C=(torch.FloatTensor(td_error))**2

        self.optimizer_C.zero_grad()
        self.optimizer_A.zero_grad()

        loss_A.backward()
        loss_C.backward()

        self.optimizer_A.step()
        self.optimizer_C.step()







if __name__== "__main__":

    MAX_EPISODE=3000
    MAX_EP_STEPS = 1000  # maximum time step in one episode
    DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    RL = A_C()
    sum_of_reward = []
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []

        while True:
            # if RENDER: env.render()

            a,log_prob, entropy = RL.choose_actions_of_A(s)

            s_, r, done, info = env.step(a)

            if done: r = -20

            track_r.append(r)

            # td_error = RL.learning_of_C(s, r, s_,done)  # gradient = grad[r + gamma * V(s_) - V(s)]
            # RL.learning_of_A(s, a, td_error,log_prob, entropy)  # true_gradient = grad[logPi(s,a) * td_error]
            RL.learning_of_C(s,r,s_,done,log_prob,entropy)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                if t % 10 ==0:
                    RL.adjust_lr()

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                sum_of_reward.append(running_reward)
                break
    path_reward = ['./data/reward/' + str('my_ac') + '/']
    folder = os.path.exists(path_reward[0])
    if not folder:
        os.makedirs(path_reward[0])
    file = open(
        path_reward[0] + '_episode_max' + str(MAX_EPISODE) + '_learning_flag' + str(MAX_EP_STEPS) + '_r_d' + str(
            0.98) + '_lr_a' + str(lr_A) + '_lr_c' + str(lr_C) + '.txt', 'w')
    file.write(str(sum_of_reward))
    file.close()

    plt.plot([i for i in range(len(sum_of_reward))], sum_of_reward)
    plt.show()
