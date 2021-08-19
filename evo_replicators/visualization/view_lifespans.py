import os
import pickle

if __name__ == "__main__":

    filepath = os.path.dirname(os.path.realpath(__file__))
    with open(filepath + "/../saves/lifespans.pkl", "rb") as f:
        lifespans = pickle.load(f)

    import matplotlib.pyplot as plt
    plt.plot(list(range(len(lifespans))), lifespans)
    plt.show()
