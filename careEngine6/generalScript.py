import pandas as pd
from MyClasses.trajectory import targetLine
from MyClasses.pathFinder import PathFinderGrids

working_directory= f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500"
ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar"
allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/provisionalAllRuns.csv")
varsigma=500

pathFinderGridsInstance = PathFinderGrids(
    working_directory= working_directory ,
    ENGINE_PATH= ENGINE_PATH,
    allRuns_csv = allRuns_csv,
    varsigma=varsigma)

##Lines terminating around 20
pathFinderGridsInstance.createSelection(name = "twenty", minH=20, maxH = 21, color = 0.6)
#pathFinderGridsInstance.checkSelection(selection = "twenty", OBS_PERIOD=100)
##Plot 5 selected lines
fig, axe,seeds_20  = pathFinderGridsInstance.plot_some_lines(selection ="twenty", window = 200, stateVar ="H", OBS_PERIOD=5)
fig.show()
#Plot grid for q100
figUPPER, params = pathFinderGridsInstance.full_gird(selectionName = "twenty", seed = seeds_20[100][0], title = "UPPER (line ending wit H around twenty)", OBS_PERIOD=100)
figUPPER.show()
#Reproduce line q100
line = targetLine(working_directory = "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500/twenty",
                  ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar",
                  seed = seeds_20[0][0], color = 0.6, name = "percentile 0")
fig = line.reproduceLine(repetitions = 15)
fig.savefig('/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twenty0lineRep.png')
fig.show()
line = targetLine(working_directory = "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500/twenty",
                  ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar",
                  seed = seeds_20[50][0], color = 0.6, name = "percentile 50")
fig = line.reproduceLine(repetitions = 15)
fig.savefig('/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twenty50lineRep.png')
fig.show()
line = targetLine(working_directory = "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500/twenty",
                  ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar",
                  seed = seeds_20[100][0], color = 0.6, name = "percentile 100")
fig = line.reproduceLine(repetitions = 15)
fig.savefig('/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twenty100lineRep.png')
fig.show()

# print(line.params)
# line.runLine()

#f = pathFinderGridsInstance.full_gird(selectionName = "twenty", seed = seeds[3], title = "3 line ending wit H around twenty)", OBS_PERIOD=100)
#f.show()
#figUPPER.savefig(f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twentyUPPERgrid.png")
#figMIDDLE, params = pathFinderGridsInstance.full_gird(selectionName = "twenty", seed = seeds[2], title = "MIDDLE (line ending wit H around twenty)", OBS_PERIOD=100)
#figMIDDLE.show()
#f = pathFinderGridsInstance.full_gird(selectionName = "twenty", seed = seeds[1], title = "1 line ending wit H around twenty)", OBS_PERIOD=100)
#f.show()
#figMIDDLE.savefig(f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twentyMIDDLEgrid.png")
#figBOTTOM, params = pathFinderGridsInstance.full_gird(selectionName = "twenty", seed = seeds[0], title = "BOTTOM (line ending wit H around twenty)", OBS_PERIOD=100)
#figBOTTOM.show()
#figBOTTOM.savefig(f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/twentyBOTTOMgrid.png")
#print(params)
#axes = pathFinderGridsInstance.grid_row(selectionName= "twenty", stateVar= "SimpleE", seed = seeds[4], axes = axes)

#pathFinderGridsInstance.createSelection(name = "fifty", minH=50, maxH = 51, color = 0.9)
#fig, axe = pathFinderGridsInstance.plot_5_lines(selection = "fifty", window = 200, stateVar = "H", OBS_PERIOD=5)
#fig.show()
#simdata = pathFinderGridsInstance.produce(stateVariables = ["H", "N", "SimpleC", "T", "SimpleE", "SimpleB"], OBS_PERIOD = 1, selectionName = "twenty", seeds = seeds)
