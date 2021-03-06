from torch.autograd import Variable
from make_env import make_env
from gym import spaces
from MADDPG_single_brain import MADDPG
import numpy as np
import torch as th
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import time
import pdb


env = make_env('simple_spread')
n_agents = len(env.world.agents)
dim_obs_list = [env.observation_space[i].shape[0] for i in range(n_agents)]

dim_act_list = []
for i in range(n_agents):
    if isinstance(env.action_space[i], spaces.MultiDiscrete):
        size = env.action_space[i].high - env.action_space[i].low + 1
        dim_act_list.append(sum(size))
    elif isinstance(env.action_space[i], spaces.Discrete):
        dim_act_list.append(env.action_space[i].n)
    else:
        print(env.action_space[i])

capacity = 30000
batch_size = 1024  # 1024

n_episode = 100000    # 20000
max_steps = 30    # 35
episodes_before_train = 50     # 50 ? Not specified in paper

snapshot_path = "/home/jadeng/Documents/snapshot/spread_single_brain/"
snapshot_name = "spread_single_brain_latest_episode_"
path = snapshot_path + snapshot_name + '800'

maddpg = MADDPG(n_agents,
                dim_obs_list,
                dim_act_list,
                batch_size,
                capacity,
                episodes_before_train,
                action_noise="Gaussian_noise",  # ou_noises
                load_models=None)               # path

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

writer = SummaryWriter()

for i_episode in range(n_episode):
    # pdb.set_trace()
    '''
    # curriculum learning
    if i_episode < 1000:
        env.set_level(0)
    elif 1000 <= i_episode < 3000:
        env.set_level(1)
    else:
        env.set_level(2)
    '''
    env.set_level(2)
    env.set_stage(1)

    obs = env.reset()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.FloatTensor(obs).type(FloatTensor)    # obs in Tensor

    total_reward = 0.0

    av_critics_grad = np.zeros((1, 6))
    av_actors_grad = np.zeros((1, 6))
    n = 0
    print('Simple Spread')
    print('Start of episode', i_episode)
    '''
    print("Target landmark for agent 0: {}, Target landmark color: {}"
          .format(env.world.agents[1].goal_b.name, env.world.agents[1].goal_b.color))
    print("Target landmark for agent 1: {}, Target landmark color: {}"
          .format(env.world.agents[0].goal_b.name, env.world.agents[0].goal_b.color))
    '''
    for t in range(max_steps):
        env.render()
        # time.sleep(0.05)

        # obs Tensor turns into Variable before feed into Actor
        obs_var = Variable(obs).type(FloatTensor)
        # pdb.set_trace()
        action = maddpg.select_action(obs_var)      # action in Variable
        action = action[0].data                     # action in Tensor
        action_np = action.cpu().numpy()            # actions in numpy array
        # convert action into list of numpy arrays
        idx = 0
        action_ls = []
        for x in dim_act_list:
            action_ls.append(action_np[idx:(idx+x)])
            idx += x
        # pdb.set_trace()
        obs_, reward, done, _ = env.step(action_ls)

        total_reward += sum(reward)
        reward = th.FloatTensor(reward).type(FloatTensor)

        obs_ = np.concatenate(obs_, 0)
        obs_ = th.FloatTensor(obs_).type(FloatTensor)

        maddpg.memory.push(obs, action, obs_, reward)  # store in Tensor

        obs = obs_

        critics_grad, actors_grad = maddpg.update_policy()

        if maddpg.episode_done > maddpg.episodes_before_train:
            av_critics_grad += np.array(critics_grad)
            av_actors_grad += np.array(actors_grad)
            n += 1

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    mean_reward = total_reward / max_steps
    print('End of Episode: %d, mean_reward = %f, total_reward = %f' % (i_episode, mean_reward, total_reward))

    # plot of reward
    writer.add_scalar('data/reward_spread', mean_reward, i_episode)

    # plot of agent0 - speaker gradient of critic net
    for i in range(6):
        writer.add_scalar('data/agent0_critic_gradient', av_critics_grad[0][i], i_episode)

    # plot of agent0 - speaker gradient of actor net
    for i in range(6):
        writer.add_scalar('data/agent0_actor_gradient', av_actors_grad[0][i], i_episode)

    '''
    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent1_critic_gradient', av_critics_grad[1][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent1_actor_gradient', av_actors_grad[1][i], i_episode)

    # plot of agent2 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent2_critic_gradient', av_critics_grad[2][i], i_episode)

    # plot of agent2 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent2_actor_gradient', av_actors_grad[2][i], i_episode)
    '''
    # to save models every 500 episodes
    if i_episode != 0 and i_episode % 500 == 0:
        print('Save models!')
        if maddpg.action_noise == "OU_noise":
            states = {'critic': maddpg.critic,
                      'actor': maddpg.actor,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critic_target': maddpg.critic_target,
                      'actor_target': maddpg.actor_target,
                      'var': maddpg.var,
                      'ou_prevs': [ou_noise.x_prev for ou_noise in maddpg.ou_noises]}
        else:
            states = {'critic': maddpg.critic,
                      'actor': maddpg.actor,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critic_target': maddpg.critic_target,
                      'actor_target': maddpg.actor_target,
                      'var': maddpg.var}
        th.save(states, snapshot_path + snapshot_name + str(i_episode))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()






















