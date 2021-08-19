import numpy
import evo_gym


class BoxingSurvivalFight(evo_gym.Env):
    def __init__(self):
        self.env = evo_gym.make("Boxing-ram-v4")
        self.action_space = self.env.action_space
        self.action_space.shape = (self.action_space.n,)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, death, _info = self.env.step(action)
        # if the opposing boxer has more points than you (referencing ram state)
        death = death or state[19] > state[18]
        state = (state.astype(numpy.float) - 128)/128
        return state, death, _info

    def render(self, mode='human'):
        self.env.render()


class BoxingSurvivalAvoidance(evo_gym.Env):
    def __init__(self):
        self.env = evo_gym.make("Boxing-ram-v4")
        self.action_space = self.env.action_space
        self.action_space.shape = (self.action_space.n,)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, death, _info = self.env.step(action)
        # if the opposing boxer lands a single punch (referencing ram state)
        death = death or state[19] > 0
        state = (state.astype(numpy.float) - 128)/128
        return state, death, _info

    def render(self, mode='human'):
        self.env.render()




