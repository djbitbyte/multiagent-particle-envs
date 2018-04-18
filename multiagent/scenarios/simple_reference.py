import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3  # 10
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            if i == 0:
                agent.size = 0.045
            elif i == 1:
                agent.size = 0.075
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, level=0):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[1].goal_a = world.agents[0]

        if level == 0:
            world.agents[0].goal_b = world.landmarks[0]
            world.agents[1].goal_b = world.landmarks[0]
        elif level == 1:
            world.agents[0].goal_b = np.random.choice([world.landmarks[0], world.landmarks[1]])
            world.agents[1].goal_b = np.random.choice([world.landmarks[0], world.landmarks[1]])
        else:
            world.agents[0].goal_b = np.random.choice(world.landmarks)  # target for the other agent
            world.agents[1].goal_b = np.random.choice(world.landmarks)

        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.5, 0.5, 0.5])
        world.agents[1].goal_a.color = world.agents[1].goal_b.color + np.array([0.5, 0.5, 0.5])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world, stage=0):
        # if agent.goal_a is None or agent.goal_b is None:
        a0 = world.agents[0]
        a1 = world.agents[1]
        if a0.goal_a is None or a0.goal_b is None or a1.goal_a is None or a1.goal_b is None:
            return 0.0

        if stage == 0:
            # squared distance and distance from listener to landmark
            # a1 being trained of navigation, and a0 being trained of communication
            dist2 = np.sum(np.square(a0.goal_a.state.p_pos - a0.goal_b.state.p_pos))
            dist = np.sqrt(dist2)
            # decide reward here
            r = -dist2 - dist
        elif stage == 1:
            # squared distance & distance between a1 and its target
            dist2_0 = np.sum(np.square(a0.goal_a.state.p_pos - a0.goal_b.state.p_pos))
            dist_0 = np.sqrt(dist2_0)
            r0 = -dist2_0 - dist_0
            # squared distance & distance between a0 and its target
            dist2_1 = np.sum(np.square(a1.goal_a.state.p_pos - a1.goal_b.state.p_pos))
            dist_1 = np.sqrt(dist2_1)
            r1 = -dist2_1 - dist_1
            # decide reward here
            r = (r0 + r1) / 2
        return r  # -dist2  # np.exp(-dist2)

    def observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

            # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
