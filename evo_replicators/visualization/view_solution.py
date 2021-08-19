import os
from evo_replicators.evolutionary_replicator import *


if __name__ == "__main__":

    filepath = os.path.dirname(os.path.realpath(__file__))
    with open(filepath + "/../saves/oldest_replicator.pkl", "rb") as f:
        replicator = pickle.load(f)

    lifespan = 0
    render = True
    replicator.network.reset()
    replicator.env_state = replicator.env.reset()

    while True:
        lifespan += 1
        if render: replicator.env.render()
        death = replicator.step()
        if death:
            print(lifespan)
            lifespan = 0
            replicator.network.reset()
            replicator.env_state = replicator.env.reset()
