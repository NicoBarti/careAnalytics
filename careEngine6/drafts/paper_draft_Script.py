from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.trajectory import Trajectory
import pandas as pd
import numpy as np

working_directory= f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500"
ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar"
#allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/provisionalAllRuns.csv")
allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/runs28_8_25_1.csv")
scaling = 'simple'
grain = 30
saveFigTo = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/figures/paper1_draft'


# Select possible trajectories for chronic care: those ending around 5
selectionName = 'five'

traj1 = Trajectory(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, seed=17522853284751,
                   selectionName=selectionName)
traj1.plotNeedEnabeler()

distances = TrajectoryDistances(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, allRuns_csv=allRuns_csv,varsigma = 500,
                        selectionName = selectionName, minH=4, maxH = 6, grain = grain, scaling=scaling)

fig, axe, percentile = distances.plot_some_lines(selection = selectionName, OBS_PERIOD = 100, window=200, stateVar="H",
                                                 q = np.arange(0, 101, step=10), color = "viridis")
fig.show()
#Explore the paramete space:
distances.set_scaling(scaling = 'multiScaling')
distances.specialPlotAndSave(root=saveFigTo)

# Explore the ENABELERS in this selection:
seeds = []
for key in percentile.keys():
    seeds.append(percentile[key][0])
lines = distances.produce(stateVariables=["SimpleE"], selectionName = "five", OBS_PERIOD = 100, seeds = seeds)
fig, axe, p = distances.plot_some_lines(lines = lines, selection = selectionName, OBS_PERIOD = 100,
                                                 window=200, stateVar="SimpleE", color = "viridis")
fig.show()

for key in percentile.keys():
    traj1 = Trajectory(working_directory = working_directory, ENGINE_PATH = ENGINE_PATH,seed = percentile[key][0], selectionName = selectionName)
    traj1.plotNeedEnabeler()