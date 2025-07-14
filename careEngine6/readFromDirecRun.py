import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os  # import os module

directory = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/sensitivity_1/H'  # set directory path
means = pd.DataFrame({})
simulationsCounter = 0
steps = 0
for entry in os.scandir(directory):
    if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
        simulationsCounter += 1
        dd= pd.read_csv(entry.path)
        means= pd.concat([means, dd.mean()], axis=1)
means = means.mean(axis=1)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = np.arange(start=0, stop=means.shape[0], step=1), y = means, alpha=0.25, color = "blue")
axe.set_title(f"Sensitivity_1. For {simulationsCounter} simulations")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel("Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_MeanH_fromJAVA.png")

directory = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/sensitivity_1/normH'  # set directory path
means = pd.DataFrame({})
simulationsCounter = 0
steps = 0
for entry in os.scandir(directory):
    if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
        simulationsCounter += 1
        dd= pd.read_csv(entry.path)
        means= pd.concat([means, dd.mean()], axis=1)
    if simulationsCounter > 300:
        break
means = means.mean(axis=1)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = np.arange(start=0, stop=means.shape[0], step=1), y = means, alpha=0.25, color = "blue")
axe.set_title(f"Sensitivity_1. For {simulationsCounter} simulations")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel("Normalized Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_NormMeanH_fromJAVA.png")

