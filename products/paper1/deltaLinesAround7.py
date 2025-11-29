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
from MyClasses.trajectoryDistances import TrajectoryDistances
import time


#Build allRuns.csv
java_subDir = '500_cap200N5000D4'
java_output = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_decimalsFixed/{java_subDir}/H_all'
collector = PathFinderCollector(java_output)
#data = collector.consolidateRuns_csv()
#sns.histplot(data=data['H'])
#plt.show()
plotter = LinePlotter()
#Explore trayectories from H500
varsigma = 500
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/paper1/500'
ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar'
data = pd.read_csv(java_output + f'/allRuns.csv')

TEN_lineGroup = LineGroup(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma,
                    OBS_PERIOD=100, policy = 'basal', selectionName= "debug")
TEN_lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min=6.95, max=7.05, csv=data)
stateVariables = ["H"]
percentile_seeds = TEN_lineGroup.find_5_traj('H', np.arange(start=0, stop=101, step=5))

policies = ["basal","H_segmented","patient_centred"]


norm = mpl.colors.Normalize(vmin=0, vmax=10)
#cmap = mpl.colormaps['viridis']

def normColorParam(param, value):
    if param == 'fixed_tau' or param == 'fixed_capN' or param == 'fixed_lambda' or param == 'fixed_eta' or param == 'fixed_capE':
        return float(value)/10
    if param == 'fixed_kappa' or param == 'fixed_psi':
        return float(value)/1
    if param == 'W':
        return float(value)/50


ColorMap = 'viridis'
plotter.setColorMAP(ColorMap)
params = ['fixed_tau', 'fixed_capN', 'fixed_lambda', 'fixed_eta', 'fixed_capE', 'fixed_kappa', 'fixed_psi', 'W']
for param in params:
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, facecolor='ghostwhite', figsize=(20, 7))
    #for policy in policies:
    for p in range(len(policies)):
        TEN_lineGroup.setPolicy(policies[p])
        for seed in percentile_seeds:
            thisLine = TEN_lineGroup.produce(stateVariables=['H'], seeds=[seed])
            plotter.setColor(normColorParam(param =param, value= TEN_lineGroup.get_paramsFromSeed(seed)[param]))
            axes[p] = plotter.plotLine(axes[p], run_data=thisLine[seed], stateVariable='H')

        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[ColorMap]), ax=axes[p], label=param)
        axes[p].set_title(f'policy={policies[p]}')
    fig.suptitle('Lines arround 7')
    fig.show()



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

c = 0
for seed in percentile_seeds:
    plotter.setColor(colors[c])
    fig, axes = plt.subplots(nrows=len(stateVariables)+1, ncols=len(policies), constrained_layout=True, facecolor='ghostwhite', figsize=(20, 10))
    for p in range(len(policies)):
        TEN_lineGroup.setPolicy(policies[p])
        percentileLine = TEN_lineGroup.produce(stateVariables=stateVariables, seeds=[seed])
        plotter.plotLine(axe = axes[0][p], run_data= percentileLine[seed], stateVariable = 'H')
        for var in range(len(stateVariables)):
            plotter.hist(data=percentileLine[seed], stateVar =stateVariables[var], axe=axes[var+1][p], window=500)
        if c == 17:
            print('d')
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
                plotter.same_limits_x(axes[row,:], fixMin = 0, fixMax = TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capN'])
            case 'SimpleE', 'Count':
                plotter.same_limits_y(axes[row, :])
                plotter.same_limits_x(axes[row, :], fixMin=0, fixMax=TEN_lineGroup.get_paramsFromSeed(seed)['fixed_capE'])
    axes[0][0].set_title("BASAL")
    axes[0][1].set_title("H-SEGMENTED")
    axes[0][2].set_title("PATIENT-CENTRED")
    fig.show()
    c = c + 1

# for i in range(6*8):
#     data = collector.consolidateRuns_csv()
#     TEN_lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min=3.95, max=4.05, csv=data)
#     plotter = LinePlotter()
#     lines = TEN_lineGroup.get_Lines()
#     fig, ax = plt.subplots(facecolor='ghostwhite', figsize=(12, 12))
#     policy = 'basal'
#     TEN_lineGroup.setPolicy(policy)
#
#     for seed in lines['seeds']:
#         thisLine = TEN_lineGroup.produce(stateVariables=['H'], seeds=[seed])
#         # plotter.setColor(colorDelta(lines.loc[seed]['fixed_delta']))
#         ax = plotter.plotLine(ax, run_data=thisLine[seed], stateVariable='H')
#
#     # norm = mpl.colors.Normalize(vmin=minDelta, vmax=maxDelta)
#     # cmap = mpl.colormaps['viridis']
#     # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#     #              ax=ax, label='delta')
#     fig.suptitle(policy)
#     #fig.show()
#     time.sleep(600)



#percentile_seeds = TEN_lineGroup.find_5_traj('H', np.arange(start=0, stop=101, step=5))




# endpoints = [[5.95,6.05], [6.95,7.05], [9.95,10.05],[15.95,16.05], [19.95,20.05], [25.95,26.05], [29.95,30.05], [35.95,36.05]]
# for i in range(5):
#     for endpoint in endpoints:
#         TEN_lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min = endpoint[0], max = endpoint[1], csv = data)
#         TEN_lineGroup.produce(stateVariables=['H', 'SimpleE'])
#         #Plot all selected lines with H_segmented policy
#         lines = TEN_lineGroup.get_Lines()
#         #fig, ax = plt.subplots(facecolor='ghostwhite', figsize=(12, 12))
#
#         for policy in policies:
#             TEN_lineGroup.setPolicy(policy)
#             for seed in lines['seeds']:
#                 thisLine = TEN_lineGroup.produce(stateVariables = ['H'], seeds=[seed])
#                 #plotter.setColor(colorDelta(lines.loc[seed]['fixed_delta']))
#                 #ax = plotter.plotLine(ax, run_data=thisLine[seed], stateVariable= 'H')


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
#lines = lines.loc[lines['fixed_delta']<2]
#minDelta = np.array(lines['fixed_delta']).min()
#maxDelta = np.array(lines['fixed_delta']).max()
#def colorDelta(value):
#    return (value-minDelta)/(maxDelta-minDelta)

# fig, ax = plt.subplots(facecolor='ghostwhite', figsize=(12, 12))
# policy = 'basal'
# TEN_lineGroup.setPolicy(policy)
#
#
# for seed in lines['seeds']:
#     thisLine = TEN_lineGroup.produce(stateVariables = ['H'], seeds=[seed])
    #plotter.setColor(colorDelta(lines.loc[seed]['fixed_delta']))
    #ax = plotter.plotLine(ax, run_data=thisLine[seed], stateVariable= 'H')

#norm = mpl.colors.Normalize(vmin=minDelta, vmax=maxDelta)
#cmap = mpl.colormaps['viridis']
#fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#              ax=ax, label='delta')
#fig.suptitle(policy)
#fig.show()

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

