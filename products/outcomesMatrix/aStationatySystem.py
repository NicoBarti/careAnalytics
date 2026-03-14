#from zipimport import alt_path_sep

from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.trajectoryDistances import TrajectoryDistances
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/stationaySystem/'

data = pd.read_csv('/Users/nicolasbarticevic/Desktop/simulationOutputs/stationaySystem/tenLine.csv')
distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                ENGINE_PATH=ENGINE_PATH,
                                varsigma=5000, selectionName='tenLine', minH=0, maxH=1000,
                                OBS_PERIOD=500, oederByWindow=1000,
                                orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                addParam={'Pi'})

#See the line in the long run
p = LinePlotter()
stateVariables=['H']
dd = distances.produce(stateVariables=stateVariables, selectionName='tenLine', seeds=[1759413572002])[1759413572002]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5), constrained_layout=True, facecolor='ghostwhite')
ax.plot(dd['windows']['0'].to_list(), dd['H'].mean(0))
rate =  (dd['H'][str(dd['windows'].index.max())].mean(0)/dd['windows'].max()).iloc[0]
fig.suptitle(f"Aprox rate: {str(rate)}")
fig.show()

##Evolution of distributions in time
dd['H'] = dd['H'].rename(dd['windows']['0'], axis='columns')
time = [1000,2000,3000,4000,5000]
stateVariables=['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T']
fig, ax = plt.subplots(ncols=len(time), nrows=len(stateVariables), figsize=(5.5*len(time),30), constrained_layout=True, facecolor='ghostwhite')
dd = distances.produce(stateVariables=stateVariables, selectionName='tenLine', seeds=[1759413572002])[1759413572002]
for t in range(0,len(time)):
    for var in range(0,len(stateVariables)):
        p.hist(data = dd, stateVar=stateVariables[var], window=dd['windows'].loc[dd['windows']['0'] == time[t]].index[0], axe = ax[var][t],
                    x_texsize = 20)
for row in range(0,len(stateVariables)):
    p.same_limits_y(ax[row])
    p.same_limits_x(ax[row])
for t in range(0,len(time)):
    ax[0][t].set_title(f"Timestep {time[t]}", size = 30)
fig.suptitle(f"Tenline", size=40)
fig.show()

##Evolution of distributions in time by subgroup
dd['H'] = dd['H'].rename(dd['windows']['0'], axis='columns')
time = [1000,2000,3000,4000,5000]
stateVariables=['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T']
fig, ax = plt.subplots(ncols=len(time), nrows=len(stateVariables), figsize=(5.5*len(time),30), constrained_layout=True, facecolor='ghostwhite')
dd = distances.produce(stateVariables=stateVariables, selectionName='tenLine', seeds=[1759413572002])[1759413572002]
window = dd['windows'].loc[dd['windows']['0'] == time[0]].index[0]
group1 = dd['H'].loc[dd['H'][str(window)] == 0].index
group2 =  dd['H'].loc[dd['H'][str(window)] > 0].index
cmap = mpl.colormaps['Set1']
colors = [cmap(1), cmap(2)]
labels = [f'H = 0 at {time[0]}', f'H > 0 at {time[0]}']
for t in range(0,len(time)):
    for var in range(0,len(stateVariables)):
        window = dd['windows'].loc[dd['windows']['0'] == time[t]].index[0]
        g1 = dd[stateVariables[var]][str(window)].loc[group1]
        g2 = dd[stateVariables[var]][str(window)].loc[group2]
        d = [g1.to_list(), g2.to_list()]
        ax[var][t].hist(d, histtype = 'bar', color=colors, label=labels)
        ax[var][t].legend()
        ax[var][t].set_xlabel(stateVariables[var], size=20)

for row in range(0,len(stateVariables)):
    p.same_limits_y(ax[row])
    p.same_limits_x(ax[row])
for t in range(0,len(time)):
    ax[0][t].set_title(f"Timestep {time[t]}", size = 30)
fig.suptitle(f"Tenline", size=40)
fig.show()

##Plotting stationaty H
time = [1000,2000,3000,4000,5000]
stateVariables=['H']
fig, ax = plt.subplots(ncols=len(time), nrows=1, figsize=(5*len(time),5), constrained_layout=True, facecolor='ghostwhite')
dd = distances.produce(stateVariables=stateVariables, selectionName='tenLine', seeds=[1759413572002])[1759413572002]
for t in range(0,len(time)):
    window = dd['windows'].loc[dd['windows']['0'] == time[t]].index[0]
    ax[t].hist(dd['H'][str(window)]/(time[t]*rate))
for t in range(0,len(time)):
    ax[t].set_title(f"Timestep {time[t]}", size = 30)
fig.suptitle(f"Tenline - H rescaled by slope", size=40)
fig.show()
