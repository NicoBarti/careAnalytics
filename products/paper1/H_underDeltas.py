import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numpy.ma.core import maximum
from MyClasses.paramPlots import ParamPlots


from MyClasses.pathFinderCollector import PathFinderCollector
#from MyClasses.pathFinder import PathFinder
#from MyClasses.trajectoryDistances import TrajectoryDistances
#from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
import seaborn as sns
from MyClasses.lineGroup import LineGroup
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
#data = pd.read_csv(java_output + f'/allRuns.csv')

dir3 = '500_fixCap200N5000Delta3'
dir4 = '500_fixCap200N5000Delta4'
dir5 = '500_fixCap200N5000Delta5'
dir6 = '500_fixCap200N5000Delta6'
dirs = [dir3, dir4, dir5, dir6]

plotter = LinePlotter()

fig, axes = plt.subplots(nrows=1, ncols=4, constrained_layout=True, facecolor='ghostwhite', figsize=(30, 7))
a=0
maxH = []
for dir in dirs:
    data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_consistentSeed/{dir}/H_all/allRuns.csv')
    axes[a].hist(data['H'], bins = 70)
    axes[a].set_title(dir)
    maxH.append(data['H'].max())
    a+=1
maximum =np.max(maxH)
for i in range(4):
    axes[i].set_xlim(left=0, right=60)
fig.show()

data = pd.read_csv(
    f'/Users/nicolasbarticevic/Desktop/simulationOutputs/pathFinderOutputs/care6_consistentSeed/{dir3}/H_all/allRuns.csv')
fig, axe = plt.subplots(nrows=1, ncols=1, constrained_layout=True, facecolor='ghostwhite', figsize=(7, 7))
axe.hist(data['H'], bins = 70)
axe.set_xlim(left = 0, right = 30)
axe.set_title(dir3)
fig.show()

fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, facecolor='ghostwhite', figsize=(30, 12))

d1 = data[(data['H'] > 1) & (data['H'] < 4)]
paramPlot = ParamPlots(data = d1, excluded= ['N', 'totalCapacity', 'fixed_delta'])
scaledVectors = paramPlot.scaleVectors()
axes[0].boxplot(scaledVectors, orientation="horizontal", tick_labels = scaledVectors.columns)
axes[0].set_title("1 < H < 4")
d2 = data[(data['H'] > 10) & (data['H'] < 15)]
paramPlot = ParamPlots(data = d2, excluded= ['N', 'totalCapacity', 'fixed_delta'])
scaledVectors = paramPlot.scaleVectors()
axes[1].boxplot(scaledVectors, orientation="horizontal", tick_labels = scaledVectors.columns)
axes[1].set_title("10 < H < 15")

d3 = data[(data['H'] > 25) & (data['H'] < 30)]
paramPlot = ParamPlots(data = d3, excluded= ['N', 'totalCapacity', 'fixed_delta'])
scaledVectors = paramPlot.scaleVectors()
axes[2].boxplot(scaledVectors, orientation="horizontal", tick_labels = scaledVectors.columns)
axes[2].set_title("25 < H < 30")
fig.suptitle("Parameters for simulations with delta = 3")
fig.show()