import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


nodes = 64


def normal_noise(dim_action):
    n = np.random.randn(dim_action)
    return Variable(th.from_numpy(n).type(th.cuda.FloatTensor))


def gumbel_sample(dim_action, eps=1e-20):
    n = np.random.uniform(size=dim_action)
    n = -np.log(-np.log(n + eps) + eps)
    return Variable(th.from_numpy(n).type(th.cuda.FloatTensor))


def gumbel_softmax(ret, dim_action, temp=1):
    ret = ret + gumbel_sample(dim_action)
    return F.softmax(ret/temp)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        self.dim_action = dim_action
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, nodes)
        self.FC2 = nn.Linear(nodes, nodes)
        self.FC3 = nn.Linear(nodes, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        result = gumbel_softmax(result, self.dim_action)
        return result


class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        # self.FC1 = nn.Linear(dim_observation, 64)       # nn.Linear(obs_dim+act_dim, 64)
        # self.FC2 = nn.Linear(64+dim_action, 64)
        self.FC1 = nn.Linear(dim_observation + dim_action, nodes)  # nn.Linear(obs_dim+act_dim, 64)
        self.FC2 = nn.Linear(nodes, nodes)
        self.FC3 = nn.Linear(nodes, 1)

    def forward(self, obs, acts):
        # result = F.relu(self.FC1(obs))
        # combined = th.cat([result, acts], 1)    # concatenate tensors in columns
        # result = F.relu(self.FC2(combined))
        combined = th.cat([obs, acts], 1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return result








'''
n = np.random.uniform(size=self.dim_act_list[i])
act += Variable(th.from_numpy(np.random.uniform(size=self.dim_act_list[i]) * self.var[i]).type(FloatTensor))

u = tf.random_uniform(tf.shape(self.logits))
return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)
'''




