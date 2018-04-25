from OrnsteinUhlenbeckActionNoise import OrnsteinUhlenbeckActionNoise as ou
from models_hra import Critic, ActorU, ActorC
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pdb



def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self,
                 n_agents,
                 dim_obs_list,
                 dim_act_list,
                 dim_act_u,
                 dim_act_c,
                 batch_size,
                 capacity,
                 episodes_before_train,
                 lr,
                 weight_decay,
                 action_noise=None,
                 load_models=None):
        dim_obs_sum = sum(dim_obs_list)
        dim_act_sum = sum(dim_act_list)

        if load_models is None:
            self.actorsU = [ActorU(dim_obs, dim_act) for (dim_obs, dim_act) in zip(dim_obs_list, dim_act_u)]
            self.actorsC = [ActorC(dim_obs, dim_act) for (dim_obs, dim_act) in zip(dim_obs_list, dim_act_c)]
            self.criticsU = [Critic(dim_obs_sum, dim_act_sum) for i in range(n_agents)]
            self.criticsC = [Critic(dim_obs_sum, dim_act_sum) for i in range(n_agents)]
            self.actorsU_target = deepcopy(self.actorsU)
            self.actorsC_target = deepcopy(self.actorsC)
            self.criticsU_target = deepcopy(self.criticsU)
            self.criticsC_target = deepcopy(self.criticsC)
            self.criticU_optimizer = [Adam(x.parameters(), lr=lr, weight_decay=weight_decay) for x in self.criticsU]   # 0.01, 0.005
            self.criticC_optimizer = [Adam(x.parameters(), lr=lr, weight_decay=weight_decay) for x in self.criticsC]   # 0.01, 0.005
            self.actorU_optimizer = [Adam(x.parameters(), lr=lr, weight_decay=weight_decay) for x in self.actorsU]     # 0.01, 0.005
            self.actorC_optimizer = [Adam(x.parameters(), lr=lr, weight_decay=weight_decay) for x in self.actorsC]     # 0.01, 0.005
            self.var = [1.0 for i in range(n_agents)]
            if action_noise == "OU_noise":
                self.ou_noises = [ou(mu=np.zeros(dim_act_list[i])) for i in range(n_agents)]
        else:
            print('Start loading models!')
            states = th.load(load_models)
            self.criticsU = states['criticsU']
            self.criticsC = states['criticsC']
            self.actorsU = states['actorsU']
            self.actorsC = states['actorsC']
            self.criticU_optimizer = states['criticU_optimizer']
            self.criticC_optimizer = states['criticC_optimizer']
            self.actorU_optimizer = states['actorU_optimizer']
            self.actorC_optimizer = states['actorC_optimizer']
            self.criticsU_target = states['criticsU_target']
            self.criticsC_target = states['criticsC_target']
            self.actorsU_target = states['actorsU_target']
            self.actorsC_target = states['actorsC_target']
            self.var = states['var']
            if action_noise == "OU_noise":
                self.ou_noises = [ou(mu=np.zeros(dim_act_list[i]), x0=states['ou_prevs'][i]) for i in range(n_agents)]
            print('Models loaded!')

        self.memory = ReplayMemory(capacity)
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.dim_obs_list = dim_obs_list
        self.dim_act_list = dim_act_list
        self.dim_act_u = dim_act_u
        self.dim_act_c = dim_act_c
        self.dim_obs_sum = dim_obs_sum
        self.dim_act_sum = dim_act_sum
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        self.clip = 50.0    # 10
        self.action_noise = action_noise

        self.GAMMA = 0.95
        self.tau = 0.01
        self.scale_reward = 0.01

        if self.use_cuda:
            for x in self.actorsU:
                x.cuda()
            for x in self.actorsC:
                x.cuda()
            for x in self.criticsU:
                x.cuda()
            for x in self.criticsC:
                x.cuda()
            for x in self.actorsU_target:
                x.cuda()
            for x in self.actorsC_target:
                x.cuda()
            for x in self.criticsU_target:
                x.cuda()
            for x in self.criticsC_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        if self.episode_done <= self.episodes_before_train:
            return None, None, None, None

        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []

        criticsU_grad = []
        criticsC_grad = []
        actorsU_grad = []
        actorsC_grad = []

        index_obs = 0
        index_act = 0
        for agent in range(self.n_agents):
            if self.episode_done > 30000:
                self.batch_size = 2048
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
            next_states_batch = Variable(th.stack(batch.next_states).type(FloatTensor))

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            # pdb.set_trace()
            ###### critic network #####
            self.criticU_optimizer[agent].zero_grad()
            self.criticC_optimizer[agent].zero_grad()
            currentU_Q = self.criticsU[agent](whole_state, whole_action)
            currentC_Q = self.criticsC[agent](whole_state, whole_action)

            idx = 0
            next_actions_ls = []
            for i in range(self.n_agents):
                next_actionU_i = self.actorsU_target[i](next_states_batch[:, idx:(idx + self.dim_obs_list[i])])
                next_actionC_i = self.actorsC_target[i](next_states_batch[:, idx:(idx + self.dim_obs_list[i])])
                next_action_i = th.cat((next_actionU_i, next_actionC_i), 1)
                next_actions_ls.append(next_action_i)
                idx += self.dim_obs_list[i]

            next_actions = th.cat(next_actions_ls, 1)
            # pdb.set_trace()
            targetU_Q = self.criticsU_target[agent](
                next_states_batch.view(-1, self.dim_obs_sum),
                next_actions.view(-1, self.dim_act_sum)
            )
            targetC_Q = self.criticsC_target[agent](
                next_states_batch.view(-1, self.dim_obs_sum),
                next_actions.view(-1, self.dim_act_sum)
            )

            # here target_Q is y_i of TD error equation
            # target_Q = (target_Q * self.GAMMA) + (reward_batch[:, agent] * self.scale_reward)
            targetU_Q = targetU_Q * self.GAMMA + reward_batch[:, agent, :1]
            targetC_Q = targetC_Q * self.GAMMA + reward_batch[:, agent, 1:]
            lossU_Q = nn.MSELoss()(currentU_Q, targetU_Q.detach())
            lossC_Q = nn.MSELoss()(currentC_Q, targetC_Q.detach())
            lossU_Q.backward()
            lossC_Q.backward()

            if self.clip is not None:
                nn.utils.clip_grad_norm(self.criticsU[agent].parameters(), self.clip)
                nn.utils.clip_grad_norm(self.criticsC[agent].parameters(), self.clip)
            self.criticU_optimizer[agent].step()
            self.criticC_optimizer[agent].step()

            ##### actor network #####
            self.actorU_optimizer[agent].zero_grad()
            self.actorC_optimizer[agent].zero_grad()
            state_i = state_batch[:, index_obs:(index_obs+self.dim_obs_list[agent])]
            index_obs += self.dim_obs_list[agent]
            actionU_i = self.actorsU[agent](state_i)
            actionC_i = self.actorsC[agent](state_i)
            action_i_U = th.cat((actionU_i, actionC_i.detach()), 1)
            action_i_C = th.cat((actionU_i.detach(), actionC_i), 1)
            acU = action_batch.clone()
            acC = action_batch.clone()
            acU[:, index_act:(index_act + self.dim_act_list[agent])] = action_i_U
            acC[:, index_act:(index_act + self.dim_act_list[agent])] = action_i_C
            whole_actionU = acU.view(self.batch_size, -1)
            whole_actionC = acC.view(self.batch_size, -1)
            index_act += self.dim_act_list[agent]

            # pdb.set_trace()
            actorU_loss = -self.criticsU[agent](whole_state, whole_actionU)
            actorC_loss = -self.criticsC[agent](whole_state, whole_actionC)

            # update actor networks
            actorU_loss = actorU_loss.mean()
            actorU_loss.backward()
            actorC_loss = actorC_loss.mean()
            actorC_loss.backward()
            if self.clip is not None:
                nn.utils.clip_grad_norm(self.actorsU[agent].parameters(), self.clip)
                nn.utils.clip_grad_norm(self.actorsC[agent].parameters(), self.clip)
            self.actorU_optimizer[agent].step()
            self.actorC_optimizer[agent].step()

            '''
            # update actor network from gradients of physical and comm loss
            loss = []
            for i in range(len(actor_loss[0])):
                loss.append(actor_loss[:, i].mean())
            loss.backward(loss)     # wrong one

            if self.clip is not None:
                nn.utils.clip_grad_norm(self.actors[agent].parameters(), self.clip)
            self.actor_optimizer[agent].step()
            '''

            # for plotting
            c_loss.append(lossU_Q)
            a_loss.append(actorU_loss)

            criticsU_agent_grad = []
            criticsC_agent_grad = []
            actorsU_agent_grad = []
            actorsC_agent_grad = []
            for x in self.criticsU[agent].parameters():
                criticsU_agent_grad.append(x.grad.data.norm(2))
                # critics_agent_grad.append(th.mean(x.grad).data[0])
            for x in self.criticsC[agent].parameters():
                criticsC_agent_grad.append(x.grad.data.norm(2))
                # critics_agent_grad.append(th.mean(x.grad).data[0])
            for x in self.actorsU[agent].parameters():
                actorsU_agent_grad.append(x.grad.data.norm(2))
                # actorsU_agent_grad.append(th.mean(x.grad).data[0])
            for x in self.actorsC[agent].parameters():
                actorsC_agent_grad.append(x.grad.data.norm(2))
                # actorsC_agent_grad.append(th.mean(x.grad).data[0])

            criticsU_grad.append(criticsU_agent_grad)
            criticsC_grad.append(criticsC_agent_grad)
            actorsU_grad.append(actorsU_agent_grad)
            actorsC_grad.append(actorsC_agent_grad)

        # update of target network
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.criticsU_target[i], self.criticsU[i], self.tau)
                soft_update(self.criticsC_target[i], self.criticsC[i], self.tau)
                soft_update(self.actorsU_target[i], self.actorsU[i], self.tau)
                soft_update(self.actorsC_target[i], self.actorsC[i], self.tau)

        return criticsU_grad, criticsC_grad, actorsU_grad, actorsC_grad

    def select_action(self, obs):   # concatenation of observations from agents
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        # pdb.set_trace()
        # obs is Variable with dimension of dim_state_sum
        obs = obs.view(-1, self.dim_obs_sum)
        actions = Variable(th.zeros(1, self.dim_act_sum)).type(FloatTensor)

        index_obs = 0
        index_act = 0
        for i in range(self.n_agents):
            sb = obs[:, index_obs:(index_obs+self.dim_obs_list[i])]
            actU = self.actorsU[i](sb)
            actC = self.actorsC[i](sb)
            act = th.cat((actU, actC), 1)
            # act = act.view(self.dim_act_list[i])

            # add exploration noise of OU process or Gaussian
            if self.action_noise == "OU_noise":
                if self.dim_act_list[i] == 5:
                    act += Variable(th.FloatTensor(self.ou_noises[i]() * self.var[i]).type(FloatTensor))
                if self.dim_act_list[i] == 8:
                    noise = th.FloatTensor(self.ou_noises[i]() * self.var[i]).type(FloatTensor)
                    noise[-3:] = th.zeros(3).type(FloatTensor)
                    act += Variable(noise)
            elif self.action_noise == "Gaussian_noise":
                if self.dim_act_list[i] == 5:
                    act += Variable(
                        th.FloatTensor(np.random.randn(self.dim_act_list[i]) * self.var[i]).type(FloatTensor))
                if self.dim_act_list[i] == 8:
                    noise = th.FloatTensor(np.random.randn(self.dim_act_list[i]) * self.var[i]).type(FloatTensor)
                    noise[-3:] = th.zeros(3).type(FloatTensor)
                    act += Variable(noise)

            # decay of action exploration
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998

            # ? remove ?
            act = th.clamp(act, -1.0, 1.0)
            actions[:, index_act:(index_act+self.dim_act_list[i])] = act

            index_obs += self.dim_obs_list[i]
            index_act += self.dim_act_list[i]

        self.steps_done += 1

        return actions


















