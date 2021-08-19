import evo_gym
import numpy as np


class PongSurvival(evo_gym.Env):
    def __init__(self):
        self.env = evo_gym.make("Pong-ram-v4")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.observation_space.shape = (3,)
        self.action_space.shape = (self.action_space.n,)

    def reset(self):
        ram = self.env.reset()
        ball_x = ram[49] - 128
        ball_y = ram[54] - 128
        player_paddle_y = ram[51] - 128
        state = np.array([player_paddle_y, ball_x, ball_y]) / 128
        return state

    def step(self, action):
        ram, death, _info = self.env.step(action)
        ball_x = ram[49] - 128
        ball_y = ram[54] - 128
        player_paddle_y = ram[51] - 128
        state = np.array([player_paddle_y, ball_x, ball_y]) / 128
        # if the opposing boxer has more points than you (referencing ram state)
        if _info["point"] == -1: death = True
        return state, death, _info

    def render(self, mode='human'):
        self.env.render()


class PongSurvivalForager(evo_gym.Env):
    def __init__(self):
        self.hunger = 0
        self.max_hunger = 500
        self.env = evo_gym.make("Pong-ram-v4")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.observation_space.shape = (3,)
        self.action_space.shape = (self.action_space.n,)

    def reset(self):
        self.hunger = 0
        ram = self.env.reset()
        ball_x = ram[49] - 128
        ball_y = ram[54] - 128
        player_paddle_y = ram[51] - 128
        state = np.array([player_paddle_y, ball_x, ball_y]) / 128
        return state

    def step(self, action):
        ram, death, _info = self.env.step(action)
        ball_x = ram[49] - 128
        ball_y = ram[54] - 128
        player_paddle_y = ram[51] - 128
        state = np.array([player_paddle_y, ball_x, ball_y]) / 128
        # if the opposing boxer has more points than you (referencing ram state)
        self.hunger += 1
        if _info["point"] == 1:
            self.hunger = 0
        elif _info["point"] == -1:
            death = True
        elif self.hunger >= self.max_hunger:
            death = True
        return state, death, _info

    def render(self, mode='human'):
        self.env.render()




