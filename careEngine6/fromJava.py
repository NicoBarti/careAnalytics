import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os  # import os module

def collector(directory):
    means = pd.DataFrame({})
    simulationsCounter = 0
    for entry in os.scandir(directory):
        if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
            simulationsCounter += 1
            dd = pd.read_csv(entry.path)
            means = pd.concat([means, dd.mean()], axis=1)
    means = means.mean(axis=1)
    return (means, simulationsCounter)

def plotOutput(data, simulationsCounter, name):
    Fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
    for row in range(data.shape[0]):
        axe.scatter(x=np.arange(start=0, stop=data.shape[1], step=1), y=data.iloc[row,:], alpha=0.1, color="blue")
    axe.scatter(x=np.arange(start=0, stop=data.shape[1], step=1), y=data.mean(), alpha=1, color="black")
    axe.set_title(f"Sensitivity_1. For {simulationsCounter} simulations")
    axe.set_xlabel("Time steps (weeks)")
    axe.set_ylabel(name)
    timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"
    Fig.show()
    Fig.savefig(f"../figures/sensitivity_1/sensitivity_1_{name}_{timestamp}_fromJAVA.png")

data, simulationsCounter = collector("../data/sensitivity_1/H")
plotOutput(data, simulationsCounter, "H")
