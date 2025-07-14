import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def normalizeH(data):
    """Normalize the mean H by its theoric maximum
    The expected mean H is varsigma*delta/52.
    According to sensitivity_1, the maximum is varsigma*10/52

    Input: a dataframe with mean H and the varsigma of the simulation it comes from
    Output: a dataframe with varsigma and normalized mean H
    """
    varsigma = []
    normH = []
    for row in range(data.shape[0]):
        time = data.iloc[row]["time"]
        varsigma.append(time)
        normH.append(data.iloc[row]["meanH"]/(time*10/52))
    return pd.DataFrame({"varsigma": varsigma, "normH": normH})

def quantile(data, quantile):
    """Compute the quantile of mean H ar each time step

    Input: a dataframe with mean H and the varsigma of the simulation it comes from
    Output: a dataframe with quantile of mean H at each step
    """
    varsigma = []
    qH = []
    for uniquetime in data["time"].unique():
        filter = data.loc[data["time"] == uniquetime]
        qu = filter["meanH"].quantile(q=quantile)
        varsigma.append(uniquetime)
        qH.append(qu)
    return pd.DataFrame({"varsigma": varsigma, "qH": qH})

def meanHruns(data):
    """Compute the mean of mean H ar each time step

    Input: a dataframe with mean H and the varsigma of the simulation it comes from
    Output: a dataframe with the mean of mean H at each step
    """
    varsigma = []
    menHrun = []
    for uniquetime in data["time"].unique():
        filter = data.loc[data["time"] == uniquetime]
        m = filter["meanH"].mean()
        varsigma.append(uniquetime)
        menHrun.append(m)
    return pd.DataFrame({"varsigma": varsigma, "meanHruns": menHrun})

def meanHrunsNORM(data):
    """Compute the normalized mean of mean H ar each time step

    Input: a dataframe with mean H and the varsigma of the simulation it comes from
    Output: a dataframe with the mean of mean H at each step
    """
    varsigma = []
    menHrun = []
    provisionalData = normalizeH(data)
    for uniquetime in provisionalData["varsigma"].unique():
        filter = provisionalData.loc[provisionalData["varsigma"] == uniquetime]
        m = filter["normH"].mean()
        varsigma.append(uniquetime)
        menHrun.append(m)
    return pd.DataFrame({"varsigma": varsigma, "meanHruns": menHrun})

d1 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_8_20.csv")
d2 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_8_32.csv")
d3 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_8_56.csv")
d4 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_9_50.csv")
d5 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_10_2.csv")
d6 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_10_19.csv")
d7 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_10_50.csv")
d8 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_11_16.csv")
d9 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_11_26.csv")
d10 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_12_15.csv")
d11 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_12_26.csv")
d12 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_12_40.csv")
d13 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_12_53.csv")
d14 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_13_5.csv")
d15 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_13_16.csv")
d16 = pd.read_csv("../figures/sensitivity_1/data/sensitivity_1_192_13_28.csv")

concat = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9, d10, d11,d12,d13,d14,d15,d16])
timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"
simulationsPerTimeWindow = concat.shape[0]/len(concat["time"].unique())

data = normalizeH(concat)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = data["varsigma"], y = data["normH"], alpha=0.25, color = "blue")
axe.set_title(f"Sensitivity_1.  {simulationsPerTimeWindow} simulations per time-window")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel("Normalized Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_NormMeanH_{timestamp}.png")

quant = 0.9
data = quantile(concat,quant)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = data["varsigma"], y = data["qH"], color = "blue")
axe.set_title(f"Sensitivity_1.  {simulationsPerTimeWindow} simulations per time-window")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel(f"Quantile {quant} of Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_Quantile{quant}MeanH_{timestamp}.png")

data = meanHruns(concat)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = data["varsigma"], y = data["meanHruns"], color = "blue")
axe.set_title(f"Sensitivity_1.  {simulationsPerTimeWindow} simulations per time-window")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel(f"MeanRuns Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_MeanRuns{quant}MeanH_{timestamp}.png")

data = meanHrunsNORM(concat)
FigMeanH, axe   = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')
axe.scatter(x = data["varsigma"], y = data["meanHruns"], color = "blue")
axe.set_title(f"Sensitivity_1.  {simulationsPerTimeWindow} simulations per time-window")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel(f"MeanRuns NORMALIZED Mean H")
FigMeanH.show()
FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_MeanRunsNORM{quant}MeanH_{timestamp}.png")