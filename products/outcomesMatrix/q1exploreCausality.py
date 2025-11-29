from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.trajectoryDistances import TrajectoryDistances
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'

def generateDistancesGroup(file: str, java_output: str, working_directory: str, collect = False) -> TrajectoryDistances:
    if not os.path.exists(java_output + f'{file}_allRuns.csv'):
        collect = True
    if collect:
        # As the seed is not being exported from ecj, a seed is assigned here. TODO: fix this in ECJ.
        # Don't collect with every run, because you'll need to re run PathFinder for every seed
        collector = PathFinderCollector(java_output)
        collector.buildRuns_ecj(file)
    data = pd.read_csv(java_output + f'{file}_allRuns.csv')
    data.rename(columns={'Fitness': "H"}, inplace=True)
    #Pick the upper quartile
    min = np.quantile(data['H'], q=.85, method='nearest')
    max = data['H'].max()
    distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                              ENGINE_PATH=ENGINE_PATH,
                                              varsigma=100, selectionName=file, minH=min, maxH=max,
                                              OBS_PERIOD=100, oederByWindow=100,
                                              orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3}, addParam={'Pi'},
                                    start = False)
    distances.set_grain(data.shape[0])
    return distances

java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design'
title = "Design - uni objective"

objectives = ['equal', 'inequal', 'effective']

#create the distance objects
for objective in objectives:
for objective in objectives:
    exec(f'dist_{objective} = generateDistancesGroup(file= \"{objective}\", java_output=\"{java_output}\", working_directory=\"{working_directory}\")')

#Generate boxplot for params
fig, axes = plt.subplots(nrows=1, ncols=len(objectives), constrained_layout=True, facecolor='ghostwhite', figsize=(5*len(objectives), 7))
for objective in range(0,len(objectives)):
    eval(f'dist_{objectives[objective]}.boxParameProfiles(cellNumber = 0, ax = axes[{objective}], fig = fig)')
    axes[objective].set_title(f"{objectives[objective]}")
fig.suptitle(title)
fig.show()

#Generate profile of Hs for best param
p = LinePlotter()
fig, axes = plt.subplots(nrows=1, ncols=4, constrained_layout=True, facecolor='ghostwhite', figsize=(20, 7))
par= "H"
for objective in range(0,len(objectives)):
    maxSeed = eval(f'dist_{objectives[objective]}.max_parameter(selection=\"{objectives[objective]}\", param=\"{par}\")')
    dd = eval(f'dist_{objectives[objective]}.produce(stateVariables=\"{par}\", selectionName=\"{objectives[objective]}\", seeds=[{maxSeed}])')
    p.hist(data = dd[int(maxSeed)], stateVar=par, window=1, axe = axes[objective],x_texsize = 15)
    axes[objective].set_title(objectives[objective], size=20)
fig.suptitle(title, size = 20)
fig.show()

