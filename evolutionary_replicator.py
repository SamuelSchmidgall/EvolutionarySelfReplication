import pickle
import evo_gym
import numpy as np
from copy import deepcopy


class RecurrentNeuralNetwork:
    """
     Simple one-hidden-layer neural network
     structure with recurrent connections on the hidden layer
    """
    def __init__(self, input_dim, output_dim, mutation_scale=0.01, hidden=32):
        """
        Neural network initialization
        :param input_dim: (int) sensory input dimensionality
        :param output_dim: (int) action output dimensionality
        :param mutation_scale: (float) weight-mutation scale
        :param hidden: (int) hidden layer dimensionality
        """
        self.hidden = hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mutation_scale = mutation_scale
        self.rw1_trace  = np.zeros((1, self.hidden))
        self.w1 = np.zeros((self.input_dim, self.hidden))  # parameter
        self.rw1  = np.zeros((self.hidden, self.hidden))   # parameter
        self.w2 = np.zeros((self.hidden, self.output_dim)) # parameter

    def reset(self):
        """ Reset recurrent trace """
        self.rw1_trace  = np.zeros((1, self.hidden))

    def forward(self, x):
        """
        Forward propagate sensory information (x)
        :param x: (np.ndarray(np.float64))
            environmental sensory information
        :return: (np.ndarray(np.float64))
            propagated action information
        """
        x = x.reshape((1, x.size))
        x = np.tanh(np.matmul(x, self.w1) +
            np.matmul(self.rw1_trace, self.rw1))
        self.rw1_trace = x
        return np.matmul(x, self.w2)

    def mutate(self):
        """
        Mutate neural network and return modified copy
        :return: (NeuralNetwork) modified/mutated copy of self
        """
        variant = deepcopy(self)
        variant.w1 += np.random.normal(
            loc=0.0, scale=variant.mutation_scale, size=variant.w1.shape)
        variant.rw1 += np.random.normal(
            loc=0.0, scale=variant.mutation_scale, size=variant.rw1.shape)
        variant.w2 += np.random.normal(
            loc=0.0, scale=variant.mutation_scale, size=variant.w2.shape)
        return variant


class Replicator:
    """ Individual self-replication structure """
    def __init__(self):
        """ Initialize replicator """
        self.lifespan = 0
        self.env = evo_gym.make("CartPole-v1")
        self.env_state = self.env.reset()
        self.network = RecurrentNeuralNetwork(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n)

    def replicate(self):
        """
        Replicate mutated copy of self
        :return: (Replicator) mutated copy
        """
        variant = deepcopy(self)
        variant.lifespan = 0
        variant.network.reset()
        variant.env_state = variant.env.reset()
        variant.network = variant.network.mutate()
        return variant


    def step(self):
        """
        Replicator's interaction with environment
        :return: (bool) whether or not replicator died
        """
        self.lifespan += 1
        action = self.network.forward(self.env_state)
        action = np.argmax(action)
        self.env_state, death, _ = self.env.step(action)
        return death


class ReplicationGrid:
    """
    1-Dimensional grid containing replicators
    -- the medium with which replication occurs
    """
    def __init__(self):
        """ Initialize grid parameters """
        self.organisms = 0
        self.grid_dim = 32
        self.mutation_prob = 1.0
        self.replication_prob = 0.1
        self.grid_range = list(range(self.grid_dim))
        self.grid = [None for _ in range(self.grid_dim)]

    def step(self):
        """
        Replicator's interaction with environment
        including replication, creation, and death
        :return: (list(Replicator)) list of (recently) dead replicators
        """
        if self.organisms == 0:
            self.organisms += 1
            self.grid[np.random.choice(self.grid_range)] = Replicator().replicate()
        death_indices = list()
        replication_indices = list()
        for _org in self.grid_range:
            if self.grid[_org] is None:
                continue
            death = self.grid[_org].step()
            if death: death_indices.append(_org)
            elif np.random.uniform(low=0.0, high=1.0) < self.replication_prob:
                replication_indices.append(_org)
        self.replicate(replication_indices)
        return self.death(death_indices)

    def death(self, death_indices):
        """
        Remove dead replicators from grid
        :param death_indices: (list(int)) list of dead replicator indices
        :return: (list(Replicator)) list of (recently) dead replicators
        """
        dead_orgs = list()
        for _org in death_indices:
            dead_orgs.append(self.grid[_org])
            self.organisms -= 1
            self.grid[_org] = None
        return dead_orgs

    def replicate(self, replication_indices):
        """
        Replicate organism indices
        :param replication_indices: (list(int))
         list of replications that are to occur
        """
        np.random.shuffle(replication_indices)
        for _org in replication_indices:
            free_neighbors = list()
            if self.grid[(_org + 1) % self.grid_dim] is None:
                free_neighbors.append((_org + 1) % self.grid_dim)
            if self.grid[(_org - 1) % self.grid_dim] is None:
                free_neighbors.append((_org - 1) % self.grid_dim)
            if len(free_neighbors) == 0:
                break
            self.organisms += 1
            np.random.shuffle(free_neighbors)
            if np.random.uniform(low=0.0, high=1.0) < self.mutation_prob:
                self.grid[free_neighbors.pop(0)] = self.grid[_org].replicate()
            else:
                self.grid[free_neighbors.pop(0)] = deepcopy(self.grid[_org])



if __name__ == "__main__":
    reward_saves = list()
    repl_grid = ReplicationGrid()
    max_lifespan, top_solution = 0, None
    for _k in range(100000):
        _dead_orgs = repl_grid.step()
        """ Save oldest solution """
        for _d_org in _dead_orgs:
            if _d_org.lifespan > max_lifespan:
                max_lifespan, top_solution = _d_org.lifespan, _d_org
                with open("oldest_solution.pkl", "wb") as f:
                    pickle.dump(_d_org, f)
        """ Organism lifespan status records """
        if (_k+1)%1000 == 0:
            org_lifespans = [_org.lifespan for _org in repl_grid.grid if _org is not None]
            if len(org_lifespans) > 0:
                avg_lifespan = sum(org_lifespans)/len(org_lifespans)
            else:
                avg_lifespan = 0
            print("Average Lifespan: {:.2f}, Num Organisms: {}".format(
                avg_lifespan, repl_grid.organisms))
            reward_saves.append(avg_lifespan)
            with open("lifespans.pkl", "wb") as f:
                pickle.dump(reward_saves, f)











