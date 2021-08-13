import numpy as np
from evo_gym import utils
from evo_gym.envs.mujoco import mujoco_env

class ReacherForagerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.iteration = 0
        self.goal = np.random.uniform(low=-.2, high=.2, size=2)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        done = False
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        dist = abs(np.linalg.norm(vec))
        if self.iteration != 0 and dist < 0.05:
            #print(dist)
            qpos = self.data.qpos #self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 0.2:
                    break
            qpos[-2:] = self.goal
            qvel = self.data.qvel #self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            self.iteration = 0
        if self.iteration > 100:
            done = True
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.iteration += 1
        return ob, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.iteration = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
