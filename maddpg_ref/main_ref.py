from torch.autograd import Variable
from make_env import make_env
from gym import spaces
from MADDPG import MADDPG
import argparse
import numpy as np
import torch as th
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import time
import pdb

time_start = time.time()

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--consistency_interval", type=int, default=10,
                    help="Number of episodes to tally communications stats over")
parser.add_argument("-l", "--load", type=str, default=None,
                    help="Path to model to load")
parser.add_argument("--snapshot_interval", type=int, default=500,
                    help="Episodes between model snapshots")
parser.add_argument("--snapshot_path", type=str, default="/home/janet/dev/snapshot/ref/",
                    help="Path to output model snapshots")
parser.add_argument("--snapshot_prefix", type=str, default="reference_latest_episode_",
                    help="Filename prefix of model snapshots")
parser.add_argument("--print_action", action="store_true")
parser.add_argument("--print_communication", action="store_true")
parser.add_argument("--memory_capacity", type=int, default=30000,
                    help="capacity for memory replay")
parser.add_argument("--batch_size", type=int, default=1024,
                    help="batch size")
parser.add_argument("--n_episode", type=int, default=10000,   # 50000,
                    help="max episodes to train")
parser.add_argument("--max_steps", type=int, default=30,
                    help="max steps to train per episode")
parser.add_argument("--episodes_before_train", type=int, default=50,
                    help="episodes that does not train but collect experiences")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="learning rate for training")
parser.add_argument("--weight_decay", type=float, default=1e-3,
                    help="L2 regularization weight decay")
parser.add_argument("--physical_channel", type=int, default=5,
                    help="physical movement channel, default as 5; alternative as 2")
parser.add_argument("--lr_decay", type=bool, default=False,
                    help="learning rate decay, default as False")
parser.add_argument("--curriculum_learning", type=str, default=None,
                    help="different curriculum learning added for training,"
                         "None as default, other options of level, stage and obs")

args = parser.parse_args()

snapshot_interval = args.snapshot_interval
snapshot_path = args.snapshot_path
snapshot_prefix = args.snapshot_prefix
load_model = args.load
consistency_interval = args.consistency_interval
print_action = args.print_action
print_communication = args.print_communication
memory_capacity = args.memory_capacity
batch_size = args.batch_size
n_episode = args.n_episode    # 20000
max_steps = args.max_steps    # 35
episodes_before_train = args.episodes_before_train     # 50 ? Not specified in paper
lr = args.learning_rate       # 0.01
weight_decay = args.weight_decay
physical_channel = args.physical_channel
lr_decay = args.lr_decay
curriculum_learning = args.curriculum_learning

# make environment
env = make_env('simple_reference',
               print_action=print_action, print_communication=print_communication)
n_agents = len(env.world.agents)
det = np.zeros(n_agents)
dim_obs_list = [env.observation_space[i].shape[0] for i in range(n_agents)]

dim_act_list = []
for i in range(n_agents):
    if isinstance(env.action_space[i], spaces.MultiDiscrete):
        size = env.action_space[i].high - env.action_space[i].low + 1
        if physical_channel == 5:
            dim_act_list.append(int(sum(size)))
        elif physical_channel == 2:
            dim_act_list.append(int(sum(size) - 3))
        else:
            AssertionError("Unexpected value: ", physical_channel)
    elif isinstance(env.action_space[i], spaces.Discrete):
        dim_act_list.append(env.action_space[i].n)
    else:
        print(env.action_space[i])

maddpg = MADDPG(n_agents,
                dim_obs_list,
                dim_act_list,
                batch_size,
                memory_capacity,
                episodes_before_train,
                lr,
                weight_decay,
                action_noise="Gaussian_noise",  # ou_noises
                load_models=load_model)         # path

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

writer = SummaryWriter()

# learning rate decay
if lr_decay:
    scheduler_critic = [StepLR(x, step_size=10000, gamma=0.1) for x in maddpg.critic_optimizer]
    scheduler_actor = [StepLR(x, step_size=10000, gamma=0.1) for x in maddpg.actor_optimizer]

for i_episode in range(n_episode):
    if lr_decay:
        for scheduler in scheduler_critic:
            scheduler.step()
        for scheduler in scheduler_actor:
            scheduler.step()

    # curriculum learning
    if curriculum_learning == "level":
        if i_episode < 1000:
            env.set_level(0)
        elif 1000 <= i_episode < 3000:
            env.set_level(1)
        else:
            env.set_level(2)
    if curriculum_learning == "stage":
        # curriculum learning on agents task
        if i_episode < 10000:
            env.set_stage(0)
        else:
            env.set_stage(1)
    if curriculum_learning == "obs":
        # curriculum learning on agents observation
        if i_episode < 3000:
            env.set_obs(0)
        elif 3000 <= i_episode < 7000:
            env.set_obs(1)
        else:
            env.set_obs(2)
    # No curriculum learning
    if curriculum_learning is None:
        env.set_level(2)
        env.set_stage(1)
        env.set_obs(2)
        # env.set_obs(3)

    # pdb.set_trace()
    obs = env.reset()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.FloatTensor(obs).type(FloatTensor)    # obs in Tensor

    total_reward = 0.0

    av_critics_grad = np.zeros((n_agents, 6))
    av_actors_grad = np.zeros((n_agents, 6))
    n = 0
    '''
    print('Simple Reference')
    print('Start of episode', i_episode)
    print("Target landmark for agent 0: {}, Target landmark color: {}"
          .format(env.world.agents[1].goal_b.name, env.world.agents[1].goal_b.color))
    print("Target landmark for agent 1: {}, Target landmark color: {}"
          .format(env.world.agents[0].goal_b.name, env.world.agents[0].goal_b.color))
    '''
    if (i_episode % consistency_interval) == 0:
        communication_mappings = np.zeros((n_agents, 3, 3))
    episode_communications = np.zeros((n_agents, 3))

    if physical_channel == 2:
        act_up = []
        act_left = []
    elif physical_channel == 5:
        act_up = []
        act_down = []
        act_left = []
        act_right = []

    for t in range(max_steps):
        env.render()
        # time.sleep(0.05)

        # obs Tensor turns into Variable before feed into Actor
        obs_var = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs_var)      # action in Variable
        action = action[0].data                     # action in Tensor
        action_np = action.cpu().numpy()            # actions in numpy array
        # convert action into list of numpy arrays
        idx = 0
        action_ls = []
        for x in dim_act_list:
            action_ls.append(action_np[idx:(idx+x)])
            idx += x

        if physical_channel == 2:
            for i in range(n_agents):
                action_ls[i] = np.insert(action_ls[i], 0, 0)
                action_ls[i] = np.insert(action_ls[i], 2, 0)
                action_ls[i] = np.insert(action_ls[i], 4, 0)

        obs_, reward, done, _ = env.step(action_ls)
        # pdb.set_trace()
        total_reward += sum(reward)
        reward = th.FloatTensor(reward).type(FloatTensor)

        # pdb.set_trace()
        comm_1 = action_ls[0][5:8].argmax()
        comm_2 = action_ls[1][5:8].argmax()
        episode_communications[0, comm_1] += 1
        episode_communications[1, comm_2] += 1

        if physical_channel == 2:
            act_up.append(action_np[0])
            act_left.append(action_np[1])
        elif physical_channel == 5:
            act_up.append(action_np[1])
            act_down.append(action_np[2])
            act_left.append(action_np[3])
            act_right.append(action_np[4])

        obs_ = np.concatenate(obs_, 0)
        obs_ = th.FloatTensor(obs_).type(FloatTensor)

        maddpg.memory.push(obs, action, obs_, reward)  # store in Tensor

        obs = obs_

        critics_grad, actors_grad = maddpg.update_policy()

        if maddpg.episode_done > maddpg.episodes_before_train:
            av_critics_grad += np.array(critics_grad)
            av_actors_grad += np.array(actors_grad)
            n += 1

    for agent_i in range(n_agents):
        for goal_i in range(3):
            if env.world.agents[agent_i].goal_b == env.world.landmarks[goal_i]:
                communication_mappings[agent_i, goal_i, :] += episode_communications[agent_i, :]

    if (i_episode % consistency_interval) == consistency_interval - 1:
        for agent_i in range(n_agents):
            string = "Agent {}: ".format(agent_i)
            normalized_agent_mapping = communication_mappings[agent_i, :, :] / np.expand_dims(
                communication_mappings[agent_i, :, :].sum(1), 1)
            writer.add_scalar('communication/agent{}_det'.format(agent_i),
                              np.linalg.det(normalized_agent_mapping),
                              i_episode)
            det[agent_i] = np.linalg.det(normalized_agent_mapping)
            for goal_i in range(3):
                mapping = communication_mappings[agent_i, goal_i, :]
                consistency = 1 if mapping.sum() == 0 else mapping.max() / mapping.sum()
                writer.add_scalar('consistency/agent{}_goal{}'.format(agent_i, goal_i), consistency, i_episode)
                string += ("{:.1f}% ".format(consistency * 100))
            # print(string)

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    mean_reward = total_reward / max_steps
    # print('End of Episode: %d, mean_reward = %f, total_reward = %f' % (i_episode, mean_reward, total_reward))

    # plot of reward
    writer.add_scalar('data/reward_ref', mean_reward, i_episode)

    # plot of histogram for actions
    if physical_channel == 2:
        writer.add_histogram("action/Up", np.array(act_up), i_episode, bins='auto')
        writer.add_histogram("action/Left", np.array(act_left), i_episode, bins='auto')
    elif physical_channel == 5:
        writer.add_histogram("action/Up", np.array(act_up), i_episode, bins='auto')
        writer.add_histogram("action/Down", np.array(act_down), i_episode, bins='auto')
        writer.add_histogram("action/Left", np.array(act_left), i_episode, bins='auto')
        writer.add_histogram("action/Right", np.array(act_right), i_episode, bins='auto')

    # plot of agent0 - speaker gradient of critic net
    for i in range(6):
        writer.add_scalar('gradient/agent0_critic_gradient', av_critics_grad[0][i], i_episode)

    # plot of agent0 - speaker gradient of actor net
    for i in range(6):
        writer.add_scalar('gradient/agent0_actor_gradient', av_actors_grad[0][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('gradient/agent1_critic_gradient', av_critics_grad[1][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('gradient/agent1_actor_gradient', av_actors_grad[1][i], i_episode)

    # to save models every N episodes
    if i_episode != 0 and i_episode % snapshot_interval == 0:
        # print('Save models!')
        if maddpg.action_noise == "OU_noise":
            states = {'critics': maddpg.critics,
                      'actors': maddpg.actors,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critics_target': maddpg.critics_target,
                      'actors_target': maddpg.actors_target,
                      'var': maddpg.var,
                      'ou_prevs': [ou_noise.x_prev for ou_noise in maddpg.ou_noises]}
        else:
            states = {'critics': maddpg.critics,
                      'actors': maddpg.actors,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critics_target': maddpg.critics_target,
                      'actors_target': maddpg.actors_target,
                      'var': maddpg.var}
        th.save(states, snapshot_path + snapshot_prefix + str(i_episode))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()

time_end = time.time()
elapsed_time = time_end - time_start
print("Time spent: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
print("Determinant of comm mapping: {}".format(det))



















