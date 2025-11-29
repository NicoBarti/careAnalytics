import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from MyClasses.pathFinderCollector import PathFinderCollector
#from MyClasses.pathFinder import PathFinder
#from MyClasses.trajectoryDistances import TrajectoryDistances
#from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
import seaborn as sns
from MyClasses.lineGroup import LineGroup
import time

#Build allRuns.csv
java_subDir = 'tenLineGroup'
java_output = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_consistentSeed/{java_subDir}/H_all'
collector = PathFinderCollector(java_output)
#collector.fixW()
#data = collector.consolidateRuns_csv()
data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_consistentSeed/{java_subDir}/H_all/allRuns.csv')
#print(f"Found {data.shape[0]} runs")
#sns.histplot(data=data['H'])
#plt.show()
plotter = LinePlotter()
#Explore trayectories from H500
varsigma = 500
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/paper1/500'
ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar'
data = pd.read_csv(java_output + f'/allRuns.csv')

TEN_lineGroup = LineGroup(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma,
                    OBS_PERIOD=100, policy = 'basal', selectionName= "tenLineGroup")
TEN_lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min=9.5, max=10.5, csv=data)
stateVariables = ['H', 'SimpleE', 'N', 'SimpleC', 'T']
percentile_seeds = TEN_lineGroup.find_5_traj('H', np.arange(start=0, stop=101, step=10))


policies = ["basal","H_segmented","patient_centred"]

#Percentile lines
colors = np.linspace(0,1,len(percentile_seeds))
plotter.setColorMAP("tab10")
fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, facecolor='ghostwhite', figsize=(20, 7))
for p in range(len(policies)):
    c = 0
    TEN_lineGroup.setPolicy(policies[p])
    percentileLines = TEN_lineGroup.produce(stateVariables=['H'], seeds=percentile_seeds)
    for line in percentileLines:
        plotter.setColor(colors[c])
        axes[p] = plotter.plotLine(axes[p], run_data=percentileLines[line], stateVariable='H', label = str(c))
        c = c + 1
    axes[p].set_title(f"Policy: {policies[p]}")
axes = plotter.same_limits_y(axes)
axes[0].legend()
fig.show()


#Detail per line
c = 0
for seed in percentileLines:

    plotter.setColor(colors[c])
    fig, axes = plt.subplots(nrows=len(stateVariables)+1, ncols=len(policies), constrained_layout=True, facecolor='ghostwhite', figsize=(20, 33))
    for p in range(len(policies)):
        TEN_lineGroup.setPolicy(policies[p])
        percentileLine = TEN_lineGroup.produce(stateVariables=stateVariables, seeds=[seed])
        plotter.plotLine(axe = axes[0][p], run_data= percentileLine[seed], stateVariable = 'H')

        for var in range(len(stateVariables)):
            plotter.hist(data=percentileLine[seed], stateVar =stateVariables[var], axe=axes[var+1][p], window=500)
    fig.suptitle(f'LINE {c} {plotter.printableParams(TEN_lineGroup.get_paramsFromSeed(seed), policy = False)}', size=25)
    # homogenize axes:
    for row in range(len(stateVariables)+1):
        xlabel, ylabel = axes[row,:][0].get_xlabel(), axes[row,:][1].get_ylabel()
        match xlabel, ylabel:
            case ('Time','H') | ('SimpleC', 'Count') | ('H', 'Count') | ('T', 'Count'):
                plotter.same_limits_y(axes[row,:])
                plotter.same_limits_x(axes[row,:])
            case 'N', 'Count':
                plotter.same_limits_y(axes[row,:])
                plotter.same_limits_x(axes[row,:], fixMin = -0.05, fixMax = TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capN'])
            case 'SimpleE', 'Count':
                plotter.same_limits_y(axes[row, :])
                plotter.same_limits_x(axes[row, :], fixMin=-0.05, fixMax=TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capE'])
    axes[0][0].set_title("BASAL")
    axes[0][1].set_title("H-SEGMENTED")
    axes[0][2].set_title("PATIENT-CENTRED")
    fig.show()
    c = c + 1