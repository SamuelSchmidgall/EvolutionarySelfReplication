import numpy
import evo_gym


class SpaceInvadersSurvival(evo_gym.Env):
    def __init__(self):
        self.env = evo_gym.make("SpaceInvaders-ram-v4")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action_space.shape = (self.action_space.n,)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, death, _info = self.env.step(action)
        state = (state.astype(numpy.float) - 128)/128
        return state, death, _info

    def render(self, mode='human'):
        self.env.render()




