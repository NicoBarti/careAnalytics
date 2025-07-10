from MyClasses.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

ENGINE_PATH="../engine/CareEngine6.jar"

c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()
params = {"PATIENT_INIT": ["sensitivity_1"], "PROVIDER_INIT": ["sensitivity_1"]}

means = []
times = []
for varsigma in np.arange(start = 53, stop = 53*15, step = 53):
    params["varsigma"] = varsigma;
    for rep in range(30):
        c.start_server()
        data = c.socket_with_model_paramGrid_2(pd.DataFrame.from_dict(params))
        H = np.array(data[0]["H"])
        means.append((H[:,1]).mean())
        times.append(varsigma)

fig, axe = plt.subplots(1, 1, figsize=(10,10), facecolor='ghostwhite')

axe.scatter(x = times, y = means)
axe.set_title("Sensitivity_1")
axe.set_xlabel("Time steps (weeks)")
axe.set_ylabel("Mean H")
timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"
fig.show()
fig.savefig(f"../figures/sensitivity_1/sensitivity_1_{timestamp}.png")
