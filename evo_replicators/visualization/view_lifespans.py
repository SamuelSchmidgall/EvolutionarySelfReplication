import os
import pickle
import matplotlib
import matplotlib.pyplot as plt


def running_average(data, avg):
    averaged_values = list()
    r_avg = [sum(data[:avg])/avg for _ in range(avg)]
    for _k in range(len(data)):
        r_avg.pop(0)
        r_avg.append(data[_k])
        averaged_values.append(sum(r_avg)/len(r_avg))
    return averaged_values

if __name__ == "__main__":

    filepath = os.path.dirname(os.path.realpath(__file__))
    with open(filepath + "/../saves/lifespans.pkl", "rb") as f:
        lifespans = pickle.load(f)

    # make the graph a bit more attractive if you want...
    lifespans = running_average(lifespans, avg=10)

    matplotlib.rcParams.update({'font.size': 12})
    plt.plot(list(range(len(lifespans))), lifespans)
    plt.title("Average Organism Lifespan over time (Boxing-survival-fight)")
    plt.xlabel("Organism number")
    plt.ylabel("Average Lifespan (all organisms)")
    plt.show()
