import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

path = "/Users/nicolasbarticevic/Desktop/simulationOutputs/isolatedSystem/java/1765550553968"
allRuns = None

for entry in os.scandir(path):
    if os.path.isfile(entry.path) and 'csv' in entry.name:
        oneFile = pd.read_csv(entry.path)
    if allRuns is None:
        allRuns = oneFile
    else:
        allRuns = pd.concat([allRuns, oneFile], axis = 0)

# fig, ax = plt.subplots()
# for index, row in allRuns.iterrows():
#     ax.plot(row['totalCapacity'], row['p100'])
#     #ax.plot(allRuns['totalCapacity'],allRuns['p100'])
# fig.show()

sns.scatterplot(x = allRuns['totalCapacity'], y = allRuns['p100'])
plt.show()