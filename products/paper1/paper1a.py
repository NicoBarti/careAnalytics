import numpy as np
import matplotlib.pyplot as plt

from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.pathFinder import PathFinder
from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.trajectory import Trajectory

def selection_lines_NeedEnabelers(varsigma, working_directory, ENGINE_PATH,selection,allRuns_csv,
                                  minH, maxH, OBS_PERIOD, orderByVariable, oederByWindow):

    distances = TrajectoryDistances(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH,
                                allRuns_csv=allRuns_csv,varsigma = varsigma,
                         selectionName = selection, minH=minH, maxH = maxH, scaling='multiScaling',
                                OBS_PERIOD = OBS_PERIOD, orderByVariable=orderByVariable, oederByWindow=oederByWindow)

    fig, axe, percentile_seeds = distances.plot_some_lines(selection = selection, window=oederByWindow,
                                                         q = np.arange(start=0, stop=101, step=5))
    fig.show()
    for pairs in percentile_seeds.values():
        line = Trajectory(working_directory=working_directory, selectionName = selection, ENGINE_PATH=ENGINE_PATH,
                  seed = pairs[0])
        line.plotNeedEnabeler(varsigma)


#Build allRuns.csv
# varsigma = 150
# window = 50
# #java_output = f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/care6_correction/{varsigma}'
java_output = f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/care6_correction/500e/SimpleE_all'
collector = PathFinderCollector(java_output)
data = collector.consolidateRuns_csv()
#
fig, ax = plt.subplots()
ax.hist(data["fixed_capE"])
fig.suptitle("Distribution of Fixed Capacity E")
fig.show()
fig, ax = plt.subplots()
ax.hist(data["SimpleE"])
fig.suptitle("Distribution of SimpleE E")
fig.show()
fig, ax = plt.subplots()
ax.hist(data["SimpleE"]/data["fixed_capE"])
fig.suptitle("Distribution of SimpleE E")
fig.show()

varsigma = 150
window = 50
java_output = f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/care6_correction/{varsigma}'
selection_lines_NeedEnabelers(varsigma=varsigma, working_directory= '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/products/paper1/500',
                              minH = 9.98, maxH = 10.02, OBS_PERIOD= 100, orderByVariable= "H", oederByWindow= 200,
                              ENGINE_PATH='/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar', selection='ten', allRuns_csv=f'{java_output}/allRuns.csv')





