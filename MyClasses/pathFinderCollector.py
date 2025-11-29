import pandas as pd
import numpy as np
import os
import random

class PathFinderCollector:
    """Handles the output from Java"""

    def __init__(self, jar_output_directory):
        self.jar_output_directory = jar_output_directory
        self.runs = None

    def buildAllRuns_csv(self):
        """Read the Hs and seeds in jar directory and put them together in allRuns.csv (PathFinder.jar)"""

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
                        thisSeeds = pd.read_csv(sub_entry.path, dtypedtype = str)
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
        """Consolidate the full model outputs: with outcomes and params together (PathFinder.jar)"""

        all = None
        this = None
        for entry in os.scandir(f"{self.jar_output_directory}"):
            if os.path.isfile(entry.path) and 'csv' in entry.name and not 'allRuns' in entry.name:
                this = pd.read_csv(entry.path, dtype = str)
                if all is None:
                    all = this
                else:
                    all = pd.concat([all, this])
        all.rename(columns ={'seed': 'seeds'}, inplace=True)
        noDups = all.drop_duplicates(subset=['seeds'])
        if noDups.shape[0] < all.shape[0]:
            print(f'(PathFinderCollector) Removing {all.shape[0] - noDups.shape[0]} runs without duplicates')
        noDups.to_csv(f"{self.jar_output_directory}/allRuns.csv", index = False)
        self.runs = noDups
        return noDups

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

    def fixW(self):
        """Change the value of W from double to int in each pathFinder output file (PathFinder.jar)"""
        this = None
        for entry in os.scandir(f"{self.jar_output_directory}"):
            if os.path.isfile(entry.path) and 'csv' in entry.name and not 'allRuns' in entry.name:
                this = pd.read_csv(entry.path, dtype={'W': int})
                this.to_csv(f'{entry.path}', index = False)

    def buildRuns_ecj(self, filename):
        """Read form ecj's stat file and saves params and fitness"""

        lines = []
        with open(f"{self.jar_output_directory}/{filename}.stat", 'r') as file:
            for line in file:
                lines.append(line)

        paramDic = {'Fitness': [], 'fixed_lambda': [], 'fixed_tau': [], 'fixed_kappa': [] , 'fixed_rho': [], 'fixed_eta': [],
                    'fixed_capN': [], 'fixed_capE': [], 'fixed_psi': [], 'N': [], 'W': [], 'fixed_delta': [], 'totalCapacity': [], 'Pi': [],
                    'seeds': [], 'varsigma': []}
        for n in range(0,len(lines)):
            if 'Fitness' in lines[n]: #Finds the begining of an evaluation
                paramDic["Fitness"].append(float(lines[n].split(" ")[1]))
                paramArray = lines[n+1].split('|') #the next line contains param values
                for param in paramArray:
                    parName = param.strip().replace(":", "").split(" ")[0]
                    if(parName in ["W", "N", "totalCapacity", "policy"]):
                        if(parName == "policy"):
                            parName = "Pi"
                            value = str(param.strip().split(" ")[1])
                        else:
                            value = int(float((param.strip().split(" ")[1])))
                    else:
                        value = float(param.strip().split(" ")[1])
                    paramDic[parName].append(value)
        #Generate random seeds if didn't get them
        if(len(paramDic["seeds"])==0):
            for i in range(0, len(paramDic["Fitness"])):
                paramDic["seeds"].append(random.randint(1000000, 100000000000))
        if (len(paramDic["varsigma"]) == 0):
            print("NO VARSIGMA GIVEN BY FILE PUTTING 100")
            for i in range(0, len(paramDic["Fitness"])):
                paramDic["varsigma"].append(100)

        allRuns = pd.DataFrame(paramDic)

        allRuns.to_csv(f"{self.jar_output_directory}/{filename}_allRuns.csv", index=False)

    def assignSeeds(self):
        """Add or change seeds to a RunFile"""



