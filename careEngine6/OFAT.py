import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os  # import os module

def collector(directory):
    means = pd.DataFrame({})
    for entry in os.scandir(directory):
        if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
            dd = pd.read_csv(entry.path)
            means = pd.concat([means, dd], axis=0)
    return (means)

def plotOutput(data, name, onlyMeans=False, finalplot = False):
    Fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
    simulationsCounter = data.shape[0]+1
    if(not onlyMeans):
        for row in range(data.shape[0]):
            axe.scatter(x=np.arange(start=1, stop=data.shape[1]+1, step=1), y=data.iloc[row,:], alpha=0.05, color="blue")

        axe.scatter(x=np.arange(start=1, stop=data.shape[1] + 1, step=1),
                    y=data.var(axis=0), alpha=1, color="red", marker="_")
    axe.scatter(x=np.arange(start=1, stop=data.shape[1]+1, step=1), y=data.mean(), alpha=1, color="black", marker = ".")

    axe.set_title(f"{simulationsCounter} repetitions")
    axe.set_xlabel("Time steps (weeks)")
    axe.set_ylabel(name)
    timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"
    addDir = ""
    if finalplot:
        timestamp="final"
        addDir = "final/"
    Fig.show()
    tag= "variation"
    if (onlyMeans):
        tag = "onlyMeans"
    Fig.savefig(f"../figures/sensitivity_1/{addDir}sensitivity_1_{name}_{timestamp}_{tag}_fromJAVA.png")

dataW = collector("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/OFAT/W/H")
plotOutput(dataW, "W",onlyMeans=True)

