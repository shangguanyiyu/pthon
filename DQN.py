import  numpy as np
import torch
import  torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import matplotlib.pyplot as plt
from collections import Counter

target_net_replace_iter=300
memory_capacity=700
batch_size=32

env=gym.make('CartPole-v1')
env=env.unwrapped
n_features=env.observation_space.shape[0]
n_actions=env.action_space.n
lossarray=[]
duelingDQN=True
class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()#进入nn.Module的init函数，继承一些最基本的 attribute
        self.lin1=nn.Linear(in_features=n_features,out_features=16)
        self.lin1.weight.data.normal_(0,0.1)
        self.lin2=nn.Linear(in_features=16,out_features=32)
        self.lin2.weight.data.normal_(0,0.1)
        self.lin3=nn.Linear(in_features=32,out_features=64)
        self.lin3.weight.data.normal_(0,0.1)
        self.lin4=nn.Linear(in_features=64,out_features=32)
        self.lin4.weight.data.normal_(0,0.1)
        self.lin5=nn.Linear(in_features=32,out_features=16)
        self.lin5.weight.data.normal_(0,0.1)


        if duelingDQN == True:
            self.V = nn.Linear(16,out_features=1) #state-value function
            self.A = nn.Linear(16,out_features=n_actions)
        else:
            self.lin6 = nn.Linear(in_features=16, out_features=n_actions)
            self.lin6.weight.data.normal_(0,0.1)




    def forward(self,x): #Module的forward里面啥都没有，让自己定义的

        # print('x=',x)
        x=F.relu(self.lin1(x)) # F的relu才可以这样
        x=F.relu(self.lin2(x))

        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))

        x = F.relu(self.lin5(x))

        if duelingDQN == True:
            V=self.V(x)
            A= self.A(x)
            return V+(A-torch.mean(A)) #为了解决输出为0的情况
        else:
            x = self.lin6(x) #输出4个action的value值
            return x
# Net2=nn.Sequential(
#     nn.Linear(in_features=n_features,out_features=16),
#     nn.ReLU(),
#     nn.Linear(in_features=16, out_features=32),
#
#     nn.ReLU(),
#     nn.Linear(in_features=32, out_features=64),
#
#     nn.ReLU(),
#     nn.Linear(in_features=64, out_features=32),
#     nn.ReLU(),
#     nn.Linear(in_features=32, out_features=16),
#     nn.ReLU(),
#     nn.Linear(in_features=16, out_features=n_actions)
# )
class DeepQ_net(object):
    def __init__(self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.95,
            replace_target_iter=300,
            memory_size=700,
            batch_size=32,
            e_greedy_increment=None,):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_counter=0
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.q_double=True

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.eval_net,self.target_net=Net(),Net()

        self.optimizer=opt.Adam(self.eval_net.parameters(),lr=self.lr)
        self.loss_function=nn.MSELoss()

    def choose_action(self,x):
        x=Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        if np.random.uniform()< self.epsilon:
            action_values=self.eval_net.forward(x)
            action=torch.max(action_values,1)[1].data.numpy()[0]
        else:
            action=np.random.randint(0,self.n_actions)
        return action

    def store_memory(self,s,a,r,s_):
        memory=np.hstack((s,[a,r],s_)) #只存储到 1x10
        index=self.memory_counter % memory_capacity
        self.memory[index,:]=memory
        self.memory_counter+=1

    def learning(self):
        if self.learn_step_counter % self.replace_target_iter ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval网络的参数传给target_net更新参数
        self.learn_step_counter+=1
        sample_index=np.random.choice(memory_capacity,batch_size) #打破相关性；随机选一批样本
        b_memory=self.memory[sample_index,:]

        b_s=Variable(torch.FloatTensor(b_memory[:,:n_features]))
        b_a=Variable(torch.LongTensor(b_memory[:,n_features:n_features+1].astype(int)))
        b_r=Variable(torch.FloatTensor(b_memory[:,n_features+1:n_features+2]))
        b_s_=Variable(torch.FloatTensor(b_memory[:,-n_features:]))

        ###-------------------DQN-start----------------------------------------

        # q_eval=self.eval_net(b_s).gather(1,b_a) #b_a这个action的value值
        #
        # q_next=self.target_net(b_s_).detach() #分离出Variable，但无需计算梯度(因为他是target_net)
        #
        # q_target=b_r+self.gamma*q_next.max(1)[0].reshape((32,1)) #  argmaxQ(s',)-->s‘中最大的a的value   Q_target= R(t+1)+yMax(S(t+1),a:参数)  DQN

        ## DQN 的 a是不用理会的，其把q_next中最大的那个值选出来就ok; 真实值是q-target（参数固定）,我们要让eval_net去拟合他，一直拟合，loss就会下降，突然q-target的参数（更先进了）变了，对应的loss就突然增大，又得开始学
        #循环往复，直到两组的参数相似为止
        ###-----------------------end---------------------------

    #----------------------------我的DDQN--start--------------------
        # ##Y=R(t+1)+yQ(s',argmaxQ(s',a;eval_net's parameter),taeget_net'parameter)
        # q_eval=self.eval_net(b_s).gather(1,b_a)
        # q_eval4next=self.eval_net(b_s_)
        #
        # target_net=self.target_net(b_s_).detach()#冻结参数的target_net
        # q_eval4next=target_net.gather(1,q_eval4next.max(1)[1].reshape(32,1))#用targe_net来求Q值
        #
        # q_target=b_r+self.gamma*q_eval4next

      ##而DDQN 在eval_net选 动作，而该动作在target_net对于未必是最大值
    #-------------------------------end-----------

    #--------------------------我的Dueling DQN---start--------
        if duelingDQN == True:

            q_next=self.target_net(b_s_).detach() #分离出Variable，但无需计算梯度(因为他是target_net)

            q_eval=self.eval_net(b_s).gather(1,b_a) #b_a这个action的value值

            q_target=b_r+self.gamma*q_next.max(1)[0].reshape((32,1))

    #-----------------------------end------------------------

        loss =self.loss_function(q_eval,q_target)
        print('loss',loss.data.numpy())
        lossarray.append(loss.data.numpy())
        self.optimizer.zero_grad()
        """
        调用backward()函数之前都要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加。
        这样逻辑的好处是，当我们的硬件限制不能使用更大的bachsize时，使用多次计算较小的bachsize的梯度平均值来代替，更方便，坏处当然是每次都要清零梯度.       
        """
        loss.backward()
        self.optimizer.step()




if __name__== "__main__":
    reward=[]
    dqn=DeepQ_net(n_actions=env.action_space.n,n_features=env.observation_space.shape[0])
    print('n_features的长度', n_features)
    i_count=[]
    ax1 = plt.subplot(2, 1, 1)
    # 第一行第二列图形
    ax2 = plt.subplot(2, 1, 2)
    # plt.ion()
    for i in range(4000):
        s=env.reset()
        count=0
        total_reward=0
        while True:
            env.render()
            a=dqn.choose_action(s)
            s_,r,done,info= env.step(a)

            x,x_dot,theta,theta_dot = s_
            r1=(env.x_threshold-abs(x))/env.x_threshold*0.7
            r2=(env.theta_threshold_radians-abs(theta)) /env.theta_threshold_radians*0.3

            r= r1+r2
            total_reward+=r

            dqn.store_memory(s,a,r,s_)
            print(str(dqn.memory_counter) + '\t'+str(i))
            count+=1

            if i >= 1000:
                dqn.learning()
                # plt.cla()
                # plt.plot(dqn.memory_counter,loss)
                # # plt.show()
            if done:

                reward.append(total_reward)
                break

            s=s_
        # # if max(Counter(i_count).values()) >= 700:
        # #     print(Counter(i_count).values())
        # #     break
        # if count >=2000:
        #     break

    plt.sca(ax1)
    plt.plot([i for i in range(len(lossarray))],lossarray)
    plt.ylabel(ylabel='loss')

    plt.sca(ax2)
    plt.plot([i for i in range(len(reward))],reward)
    plt.ylabel(ylabel='reward')

    plt.savefig('./reward.png')  #在show之前
    plt.show()






