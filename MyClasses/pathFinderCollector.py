import pandas as pd
import numpy as np
import os

class PathFinderCollector:
    """Handles the output from PathFinder.jar"""

    def __init__(self, jar_output_directory):
        self.jar_output_directory = jar_output_directory
        self.runs = None

    def buildAllRuns_csv(self):
        """Read the Hs and seeds in jar directory and put them together in allRuns.csv"""

        allRuns = None
        onlySeeds = None
        for entry in os.scandir(f"{self.jar_output_directory}/H"):
            if os.path.isfile(entry.path) and 'csv' in entry.name and 'pathFinder' in entry.name:
                thisSeeds = None
                foundSeeds = False
                pathFinderBatch = entry.name.split('.')[0].split('_')[1]
                pathFinderTimestamp= entry.name.split('.')[0].split('_')[2]
                thisHs = pd.read_csv(entry.path)
                for sub_entry in os.scandir(f"{self.jar_output_directory}/seeds"):
                    if os.path.isfile(sub_entry.path) and 'csv' in sub_entry.name and 'pathFinder' in sub_entry.name and pathFinderTimestamp == sub_entry.name.split('.')[0].split('_')[2] and pathFinderBatch == sub_entry.name.split('.')[0].split('_')[1]:
                        thisSeeds = pd.read_csv(sub_entry.path)
                        thisSeeds.rename(columns={'H': 'seeds'}, inplace=True)
                        foundSeeds = True
                        break
                if foundSeeds:
                    if allRuns is None:
                        allRuns = pd.concat([thisHs, thisSeeds], axis = 1)
                    else:
                        allRuns = pd.concat([allRuns, pd.concat([thisHs, thisSeeds], axis = 1)], axis = 0)
                    if onlySeeds is None:
                        onlySeeds = thisSeeds
                    else:
                        onlySeeds = pd.concat([onlySeeds, thisSeeds], axis = 0)
                else:
                    print(f'No seeds found for {entry.name}')
        allRuns.to_csv(f"{self.jar_output_directory}/allRuns.csv", index = False)
        onlySeeds.to_csv(f"{self.jar_output_directory}/onlySeeds.csv", index = False, header = False)
        return

    def consolidateRuns_csv(self):
        """Consolidate the full model outputs: with outcomes and params together"""

        all = None
        this = None
        for entry in os.scandir(f"{self.jar_output_directory}"):
            if os.path.isfile(entry.path) and 'csv' in entry.name and not 'allRuns' in entry.name:
                this = pd.read_csv(entry.path)
                if all is None:
                    all = this
                else:
                    all = pd.concat([all, this])
        all.rename(columns ={'seed': 'seeds'}, inplace=True)
        all.to_csv(f"{self.jar_output_directory}/allRuns.csv", index = False)
        self.runs = all
        return all

    def filterRunsEqual(self, params):
        """Filter the runs according to the given params"""

        if self.runs is None:
            print('No runs found')
            return self.runs
        else:
            filter = self.runs
            for param in params:
                filter = filter[filter[param] == params[param]]
            return filter


