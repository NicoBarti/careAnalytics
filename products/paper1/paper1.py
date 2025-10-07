from turtledemo.penrose import start

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from MyClasses.pathFinderCollector import PathFinderCollector
#from MyClasses.pathFinder import PathFinder
#from MyClasses.trajectoryDistances import TrajectoryDistances
#from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
#import seaborn as sns
from MyClasses.lineGroup import LineGroup

#Build allRuns.csv
java_output = '/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_consistentSeed/500/H_all'
collector = PathFinderCollector(java_output)
data = collector.consolidateRuns_csv()
print(f"Found {data.shape[0]} runs")
#sns.histplot(data=data['H'])
#plt.show()
#Explore trayectories from H500
varsigma = 500
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/paper1/500'
ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar'

TEN_lineGroup = LineGroup(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma,
                      OBS_PERIOD=100, policy = 'basal', selectionName= 'tenLineGroup')
TEN_lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min = 9.5, max = 10.5, csv = data)
#TEN_lineGroup.produce(stateVariables=['H', 'SimpleE'])

#percentile_seeds = TEN_lineGroup.find_5_traj('H', np.arange(start=0, stop=101, step=5))

plotter = LinePlotter()


policies = ["basal","H_segmented","patient_centred"]
stateVariables = ["H"]


#Tewaking a line
def teakline(seed, params, ax, policy, color = 1):
    TEN_lineGroup.tweakLine(seed = seed, params=params)
    TEN_lineGroup.setPolicy(policy )
    data = TEN_lineGroup.retrieve_run_model(stateVariables=['H'], seed = seed, save=False)[seed]
    plotter.setColor(color)
    plotter.plotLine(axe = ax, run_data=data, stateVariable='H')
    fig.suptitle(plotter.printableParams(TEN_lineGroup.get_paramsFromSeed(seed), policy = False))
    TEN_lineGroup.untweakLine(seed = seed)
    return ax

## Tweak this line:
# seed = 1759243020749
# newValues = np.linspace(start=0.16, stop=1.5, num = 10)
# param, policy = 'fixed_lambda', 'basal'
# colors = np.linspace(start=0, stop=1, num=len(newValues))
# fig, ax = plt.subplots(facecolor='ghostwhite', figsize=(12, 12))
# for newValue, color in zip(newValues, colors):
#     ax=teakline(seed=seed, params={param: newValue}, ax=ax, color=color, policy = policy)
# cmap = mpl.colormaps['viridis']
# norm = mpl.colors.Normalize(vmin=np.min(newValues), vmax=np.max(newValues))
# fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#              ax=ax, label=param)
# fig.suptitle(f'Tweaking {param} with {policy} policy')
# fig.show()
# print('o')

#stateVariables = ["H", "N", "SimpleE", "SimpleC", "T"]

#Plot all selected lines with H_segmented policy
lines = TEN_lineGroup.get_Lines()
fig, ax = plt.subplots(facecolor='ghostwhite', figsize=(12, 12))
for seed in lines['seeds']:
    TEN_lineGroup.setPolicy('H_segmented')
    thisLine = TEN_lineGroup.produce(stateVariables = ['H'], seeds=[seed])
    ax = plotter.plotLine(ax, run_data=thisLine[seed], stateVariable= 'H')
fig.show()

# Percentile lines
# colors = np.linspace(0,1,len(percentile_seeds))
# plotter.setColorMAP("tab10")
# fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, facecolor='ghostwhite', figsize=(20, 7))
# for p in range(len(policies)):
#     c = 0
#     TEN_lineGroup.setPolicy(policies[p])
#     percentileLines = TEN_lineGroup.produce(stateVariables=['H', 'SimpleE'], seeds=percentile_seeds)
#     for line in percentileLines:
#         plotter.setColor(colors[c])
#         axes[p] = plotter.plotLine(axes[p], run_data=percentileLines[line], stateVariable='H', label = str(c))
#         c = c + 1
#     axes[p].set_title(f"Policy: {policies[p]}")
# axes = plotter.same_limits_y(axes)
# axes[0].legend()
# fig.show()

#Detail per line
c = 0
# for seed in percentileLines:
#     plotter.setColor(colors[c])
#     fig, axes = plt.subplots(nrows=len(stateVariables)+1, ncols=len(policies), constrained_layout=True, facecolor='ghostwhite', figsize=(20, 33))
#     for p in range(len(policies)):
#         TEN_lineGroup.setPolicy(policies[p])
#         percentileLine = TEN_lineGroup.produce(stateVariables=stateVariables, seeds=[seed])
#         plotter.plotLine(axe = axes[0][p], run_data= percentileLine[seed], stateVariable = 'H')
#         for var in range(len(stateVariables)):
#             plotter.hist(data=percentileLine[seed], stateVar =stateVariables[var], axe=axes[var+1][p], window=500)
#     fig.suptitle(f'LINE {c} {plotter.printableParams(TEN_lineGroup.get_paramsFromSeed(seed), policy = False)}', size=25)
#     # homogenize axes:
#     for row in range(len(stateVariables)+1):
#         xlabel, ylabel = axes[row,:][0].get_xlabel(), axes[row,:][1].get_ylabel()
#         match xlabel, ylabel:
#             case ('Time','H') | ('SimpleC', 'Count') | ('H', 'Count') | ('T', 'Count'):
#                 plotter.same_limits_y(axes[row,:])
#                 plotter.same_limits_x(axes[row,:])
#             case 'N', 'Count':
#                 plotter.same_limits_y(axes[row,:])
#                 plotter.same_limits_x(axes[row,:], fixMin = 0, fixMax = TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capN'])
#             case 'SimpleE', 'Count':
#                 plotter.same_limits_y(axes[row, :])
#                 plotter.same_limits_x(axes[row, :], fixMin=0, fixMax=TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capE'])
#     axes[0][0].set_title("BASAL")
#     axes[0][1].set_title("H-SEGMENTED")
#     axes[0][2].set_title("PATIENT-CENTRED")
#     fig.show()
#     c = c + 1



