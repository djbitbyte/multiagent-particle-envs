from torch.autograd import Variable
from make_env import make_env
from gym import spaces
from MADDPG import MADDPG
import argparse
import numpy as np
import torch as th
import time


# make environment
env = make_env('simple_reference',
               print_action=True, print_communication=True)
n_agents = len(env.world.agents)
dim_obs_list = [env.observation_space[i].shape[0] for i in range(n_agents)]

print("DEBUG: Hardcoding actions to remove redundant values")
dim_act_list = [5] * n_agents   # [Up/Down, right/left, 3 communication values]


path = './snapshot/reference_latest_episode_22500'
maddpg = MADDPG(n_agents,
                dim_obs_list,
                dim_act_list,
                1,
                10000,
                1,
                0.001,
                0.001,
                action_noise=None,
                load_models=path,
                validating=True)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor


for i_episode in range(100):
    env.set_level(2)
    env.set_stage(1)
    env.set_obs(2)

    obs = env.reset()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.FloatTensor(obs).type(FloatTensor)    # obs in Tensor

    total_reward = 0.0

    '''
    print('Simple Reference')
    print('Start of episode', i_episode)
    print("Target landmark for agent 0: {}, Target landmark color: {}"
          .format(env.world.agents[1].goal_b.name, env.world.agents[1].goal_b.color))
    print("Target landmark for agent 1: {}, Target landmark color: {}"
          .format(env.world.agents[0].goal_b.name, env.world.agents[0].goal_b.color))
    '''
    max_steps = 30
    for i in range(max_steps):
        env.render()
        # obs Tensor turns into Variable before feed into Actor
        obs_var = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs_var)      # action in Variable
        action = action[0].data                     # action in Tensor
        action_np = action.cpu().numpy()            # actions in numpy array
        # convert action into list of numpy arrays
        idx = 0
        action_ls = []

        # NOTE: Hardcoding unnecessary actions
        for x in dim_act_list:
            agent_action = action_np[idx:(idx+x)]
            hardcoded_action = [0, agent_action[0], 0, agent_action[1], 0] + list(agent_action[2:])
            action_ls.append(np.array(hardcoded_action))
            idx += x
        obs_, reward, done, _ = env.step(action_ls)
        total_reward += sum(reward)

        obs_ = np.concatenate(obs_, 0)
        obs_ = th.FloatTensor(obs_).type(FloatTensor)

        obs = obs_
        time.sleep(0.05)

    print("Episode {} finished. Reward={}".format(i_episode, total_reward))
