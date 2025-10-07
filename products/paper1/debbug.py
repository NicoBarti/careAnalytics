import numpy as np

from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.pathFinder import PathFinder
from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.trajectory import Trajectory

#Build allRuns.csv
java_output = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/care6_correction'
collector = PathFinderCollector(java_output)
#collector.buildAllRuns_csv()

#Explore trayectories from H500
working_directory = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/products/paper1/500'
ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar'
pathFinder = PathFinder(working_directory=working_directory,allRuns_csv=f'{java_output}/allRuns.csv',
                        ENGINE_PATH=ENGINE_PATH, filterW_capacity = False)

selection = 'ten'
distances = TrajectoryDistances(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, allRuns_csv=f'{java_output}/allRuns.csv',varsigma = 500,
                         selectionName = selection, minH=9.52, maxH = 9.55, grain = 30, scaling='multiScaling')

fig, axe, percentile_seeds = distances.plot_some_lines(selection = selection, window=200, OBS_PERIOD = 100
                                                        , q = np.arange(start=0, stop=101, step=10))
fig.show()
for pairs in percentile_seeds.values():
    line = Trajectory(working_directory=working_directory, selectionName = selection, ENGINE_PATH=ENGINE_PATH,
                  seed = pairs[0])
    line.plotNeedEnabeler()