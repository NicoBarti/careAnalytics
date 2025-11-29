from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.pathFinderCollector import PathFinderCollector
import pandas as pd
import numpy as np
import os

class EcjHandler:
    """Handles ECJ outputs (.stats)
    INPUT: file, the .stats file
    java_output: the directory whith .stats files
    working_directory: the directory where the Python runs and files will be stored
    ENGINE_PATH: the path for care engine
    collect: if false, does not construct the .csv, resuses previos, preserving the seeds
    OBS_PERIOD: the number of windows
    quantile: the simulations that will be part of the DistancesGroup, need to be run from ENGIN or recovered from working directory
    norms: adjust the max of some parameters
    addParam: for params not contained in default DistanceGroup
    """
    def __init__(self, file: str, java_output: str, working_directory: str, ENGINE_PATH:str,  collect=False,
                               OBS_PERIOD=100, quantlile = .85, norms = {"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                 addParam={'Pi'}):
        self.file = file
        self.java_output = java_output
        self.working_directory = working_directory
        self.collect = collect
        self.OBS_PERIOD = OBS_PERIOD
        self.quantlile = quantlile
        self.ENGINE_PATH = ENGINE_PATH
        self.norms = norms
        self.addParam = addParam

    def generateDistancesGroup(self) -> TrajectoryDistances:
        collect = self.collect
        if not os.path.exists(self.java_output + f'{self.file}_allRuns.csv'):
            collect = True
        if collect: #build the csv from .stats, and creates a seed
            collector = PathFinderCollector(self.java_output)
            collector.buildRuns_ecj(self.file)
        data = pd.read_csv(self.java_output + f'{self.file}_allRuns.csv')
        data.rename(columns={'Fitness': "H"}, inplace=True)
        # Pick the upper quartile
        min = np.quantile(data['H'], q=self.quantlile, method='nearest')
        max = data['H'].max()
        distances = TrajectoryDistances(working_directory=self.working_directory, allRuns_csv=data,
                                        ENGINE_PATH=self.ENGINE_PATH,
                                        varsigma=100, selectionName=self.file, minH=min, maxH=max,
                                        OBS_PERIOD=100, oederByWindow=100,
                                        orderByVariable='H', norms=self.norms, addParam=self.addParam,
                                        start=False)
        distances.set_grain(data.shape[0])
        return distances


