import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb


nodes = 64


def normal_noise(dim_action):
    n = np.random.randn(dim_action)
    return Variable(th.from_numpy(n).type(th.cuda.FloatTensor))


def gumbel_sample(dim_action, eps=1e-20):
    n = np.random.uniform(size=dim_action)
    n = -np.log(-np.log(n + eps) + eps)
    return Variable(th.from_numpy(n).type(th.cuda.FloatTensor))


def gumbel_softmax(ret, dim_action, temp=1):
    # pdb.set_trace()
    ret = ret + gumbel_sample(dim_action)
    return F.softmax(ret/temp)


class ActorU(nn.Module):
    def __init__(self, dim_observation, dim_action_u):
        self.dim_action_u = dim_action_u
        super(ActorU, self).__init__()
        self.FC1 = nn.Linear(dim_observation, nodes)
        self.FC2 = nn.Linear(nodes, nodes)
        self.FC3 = nn.Linear(nodes, dim_action_u)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        # print("Real action result: {}".format(result))
        result = F.tanh(result)     # tanh for physical action
        # print("Tanh action result: {}".format(result))
        return result


class ActorC(nn.Module):
    def __init__(self, dim_observation, dim_action_c):
        self.dim_action_c = dim_action_c
        super(ActorC, self).__init__()
        self.FC1 = nn.Linear(dim_observation, nodes)
        self.FC2 = nn.Linear(nodes, nodes)
        self.FC3 = nn.Linear(nodes, dim_action_c)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        result = gumbel_softmax(result, self.dim_action_c)    # gumbel_softmax for comm action
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














