import pandas as pd
import os  # import os module
import matplotlib.pyplot as plt
import numpy as np
from MyClasses.client import *
from MyClasses.plotting import *

def collector2(directory_seed, directory_data):
    seeds = []
    data = []
    for entry in os.scandir(directory_seed):
        if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
            readSeeds =pd.read_csv(entry.path)["H"]
            for seed in readSeeds:
                seeds.append(seed)
            runNumber = entry.name.split("_")[1].split("pathFinder")[0]
            for entry2 in os.scandir(directory_data):
                if entry2.is_file() and "csv" in entry.name.split(".") and runNumber in entry2.name.split("_")[1].split("pathFinder"):
                    readData = pd.read_csv(entry2.path)["H"]
                    for result in readData:
                        data.append(result)
                    break
    return (pd.DataFrame({"seeds": seeds, "data": data}))

def retreiver(seed, directory):
    #retrivers = [False, False, False]
    #B and C have w as dimension, so I'm not dealing with it now yet
    retrivers = [False]
    #types = ["C", "B", "H"]
    types = ["H"]
    #simdata = {"B": []}
    simdata = {"H": []}
    for i in range(0,1):
        for entry in os.scandir(f"{directory}/{types[i]}"):
            if str(seed) in entry.name.split("."):
                simdata[types[i]] = pd.read_csv(entry.path)
                retrivers[i] = True
                break
    return (retrivers[0], simdata)

def retreive(selection,runRow, directory):
    #for runRow in range(0,selection.shape[0]):
        seed = selection["seeds"].iloc[runRow]
        dat = selection["data"].iloc[runRow]
        retreived, simdata = retreiver(directory = directory, seed=seed)

        if not retreived:
            runerdata = c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame({"varsigma": [500],"seed": [seed], "OBS_PERIOD": [5]}))
            simdata = {"H": []}
            c.start_server()
            for type in ["H"]: #,["H "B", "C"]:
                simdata[type] = pd.DataFrame(runerdata[0][type], columns=runerdata[0]["windows"])
                simdata[type].to_csv(f'{directory}/{type}/{seed}.csv')

        if simdata["H"].iloc[:,-1].mean().round(3) != dat.round(3):
            raise ("Retreived results don't match in H")
        return(simdata)

def retreiveForGrid(selection,runRow, directory, OBS_PERIOD = 100):
    """Retreive from disk or from engine the distributions for the grid
        H, N, C, T, E, B in rows = 6 rows"""
    seed = selection["seeds"].iloc[runRow]

def find_5_traj(data, window = '200'):
    """Finnd the trajectories with percentil 0, .25, .50, .75, .100 in data for the given window

    Input: data, a dictionaty of lineNumber: observations; window a string with the name of the timestep of interes
    Output: an array with the lineNumber for the respective percentil"""

    h_200 = []
    for trajectory in data:
        h_200.append(data[trajectory].loc[str(window)])
    h_200 = np.array(h_200)
    qs = np.percentile(h_200, q= [0,25, 50, 75, 100], axis=0, method="nearest")
    lineNumers=[]
    for trajectory in data:
        if data[trajectory].loc[str(window)] in qs:
            lineNumers.append(trajectory)
    return lineNumers

def plot_5_lines(data,type, window = '200'):
    """Plot the lines for the percentiles 0, 0.25, 0.5, 0.75, 1 at time window window
    Input: data, the dictionary with all the data for each type.
    type: (str) the data type
    Output: (fig, axe)"""

    typeData = data[type]
    lineNumers = find_5_traj(typeData)
    print(lineNumers)
    fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
    for line in lineNumers:
        axe.plot([int(x) for x in typeData[line].index.to_list()], typeData[line].to_list(), color = runGrups[type][3][0], alpha = 1)
    return fig, axe

def states_Grid(selection):
    """Plots the grid:
    H, N, C, T, E, and B in rows = 6 rows,
    windows 100, 200, 300, 400 ,500 in columns = 5 cols."""

    fig, axe = plt.subplots(6, 5, figsize=(10, 10), facecolor='ghostwhite')
    print(selection)


allruns = pd.read_csv("/data/pathFinder/provisionalAllRuns.csv")
print(f"Number of runs: {allruns.shape}")
fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
axe.hist(x=allruns["data"])
axe.set_xlabel("H", size = 15)
axe.set_ylabel("Number of simulations", size = 15)
#fig.show()
#fig.savefig('/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/hist.png')

ENGINE_PATH = "/engine/CareEngine6_socket.jar"
timestamp = f"{time.gmtime().tm_yday}_{time.gmtime().tm_hour}_{time.gmtime().tm_min}"
cmap = mpl.colormaps['viridis']
runGrups = {
            #"one": ["provisional_HBC_outputs_1", 1,1.05, cmap([0.1])],
            #"ten": ["provisional_HBC_outputs_10", 10,11, cmap([0.3])],
            "twenty": ["provisional_HBC_outputs_20", 20,21, cmap([0.6])],
            #"fifty": ["provisional_HBC_outputs_50", 50,50.5, cmap([0.9])]
}

c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()

types = {}
for type in runGrups:
    ## First, select a small group of runs that concentrate on an end point
    currentGroup = runGrups[type]
    directory=f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/{currentGroup[0]}"
    selection = allruns.loc[(allruns["data"] >= currentGroup[1]) & (allruns["data"] <= currentGroup[2])]
    print(f"Selection contains of {selection.shape[0]} runs")
    ## Then for each selected runs, retreive the complete simulations
    selectedLRunsAverages = {}
    for runRow in range(0,selection.shape[0]):
        simdata = retreive(selection = selection, directory = directory, runRow = runRow)
        plotlines = simdata["H"].mean(0)[1:]
        selectedLRunsAverages[runRow] = plotlines
    types[type] = selectedLRunsAverages


fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')

for type in runGrups:
    fig, axe = plot_5_lines(data= types, type = type)
    fig.suptitle(f"Five selected lines that ended with H around {type}", size = 20)
    fig.show()
    fig.savefig(f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/fiveLines_{type}.png')

