from MyClasses.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

ENGINE_PATH="../engine/CareEngine6.jar"

c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()

params = []
times = []
repetitions = 50

varsigmaGrid = np.arange(start = 52, stop = 53*20, step = 52)
varsigmas = []
for i in range(repetitions):
    varsigmas = np.concatenate((varsigmas, varsigmaGrid))
for varsigma in varsigmas:
    params.append({"PATIENT_INIT": "sensitivity_1", "PROVIDER_INIT": "sensitivity_1", "varsigma": varsigma})
data = c.socket_with_model_paramGrid_2(pd.DataFrame.from_dict(params))

meansH = []
variancesH = []
meansB = []
variancesB = []
meansC = []
variancesC = []

for row in data:
    H = np.array(data[row]["H"])
    meansH.append((H[:,1]).mean())
    variancesH.append((H[:,1]).var())
    B = np.array(data[row]["B"])
    meansB.append((H[:,1]).mean())
    variancesB.append((H[:,1]).var())
    C = np.array(data[row]["B"])
    meansC.append((H[:,1]).mean())
    variancesC.append((H[:,1]).var())
    times.append(varsigmas[row])
timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"

d = pd.DataFrame({"time" : times, "meanH" : meansH, "meanB" : meansB, "meanC" : meansC,
                  "varH": variancesH, "varB" : variancesB, "varC" : variancesC})
pd.DataFrame(d).to_csv(f"../figures/sensitivity_1/data/sensitivity_1_{timestamp}.csv")

# FigMeanH, axe   = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = meansH, alpha=0.5, color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Mean H")
# FigMeanH.show()
# FigMeanH.savefig(f"../figures/sensitivity_1/sensitivity_1_meanH_{timestamp}.png")
#
# FigVarH, axe = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = variancesH, alpha=0.5,  color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Var H")
# FigVarH.show()
# FigVarH.savefig(f"../figures/sensitivity_1/sensitivity_1_varH_{timestamp}.png")
#
# FigMeanB, axe   = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = meansB, alpha=0.5, color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Mean B")
# FigMeanB.show()
# FigMeanB.savefig(f"../figures/sensitivity_1/sensitivity_1_meanH_{timestamp}.png")
#
# FigVarB, axe = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = variancesB, alpha=0.5,  color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Var B")
# FigVarB.show()
# FigVarB.savefig(f"../figures/sensitivity_1/sensitivity_1_varH_{timestamp}.png")
#
# FigMeanC, axe   = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = meansC, alpha=0.5, color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Mean C")
# FigMeanC.show()
# FigMeanC.savefig(f"../figures/sensitivity_1/sensitivity_1_meanH_{timestamp}.png")
#
# FigVarC, axe = plt.subplots(1, 1, figsize=(7,7), facecolor='ghostwhite')
# axe.scatter(x = times, y = variancesC, alpha=0.5,  color = "blue")
# axe.set_title("Sensitivity_1")
# axe.set_xlabel("Time steps (weeks)")
# axe.set_ylabel("Var B")
# FigVarC.show()
# FigVarC.savefig(f"../figures/sensitivity_1/sensitivity_1_varH_{timestamp}.png")

print("end")