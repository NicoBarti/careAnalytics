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
from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.paramPlots import ParamPlots

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

lineGroup = LineGroup(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma,
                    OBS_PERIOD=100, policy = 'basal', selectionName= "debug")
lineGroup.createSelectionFromCSV_pathfinder(stateVariable='H', min=6.95, max=7.05, csv=data)


evaluations = pd.read_csv(f'{working_directory}/sevenLines/evaluations.csv')
print(evaluations)


seeds = evaluations['seeds']
params = lineGroup.get_paramsFromSeed(seeds)

paramPlot = ParamPlots(data = params, excluded= ['N', 'totalCapacity', 'fixed_delta'])
scaledVectors = paramPlot.scaleVectors()
fig, ax = plt.subplots()
ax.boxplot(scaledVectors, orientation="horizontal", tick_labels = scaledVectors.columns)
fig.show()
