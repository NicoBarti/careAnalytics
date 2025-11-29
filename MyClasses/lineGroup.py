import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from numpy import dtype

from MyClasses.client import Client
import numpy as np
import json

#from careEngine6.drafts.paper_draft_Script import selectionName


class LineGroup:
    """Recover and store runs (lines)"""

## Files names:
    ## varsigma_policy_VAR_SEED_OBSPERIOD
    ## f"var{varsigma}_pi{policy}_{H}_{SEED}_OBS{OBSPERIOD}"
## Whitin a LineGroup instance, the SEED is the identifier of a run. There must not be repeated seeds. To run a seed with
    ## different OBS_PERIOD or policy or varsicma, you need create another LineGorup instance.

    def __init__(self, working_directory: object, ENGINE_PATH: object, varsigma: object, OBS_PERIOD: object, policy: object, selectionName: object) -> None:
        ##allRuns_csv is a direct output from Java with the end values of H (culmn data) and the seed for each run.
        self.working_directory = working_directory
        self.varsigma = varsigma
        self.policy = policy
        self.ENGINE_PATH = ENGINE_PATH
        self.c = Client(ENGINE_PATH=ENGINE_PATH)
        self.c.start_server()
        self.OBS_PERIOD = OBS_PERIOD
        self.selectionName = selectionName
        self.makedir()
        self.selection = None

    def makedir(self):
        """Create a directory for every selection"""
        if not os.path.exists(self.working_directory + "/" + self.selectionName):
            os.makedirs(self.working_directory + "/" + self.selectionName)

    def createSelectionFromCSV_pathfinder(self, stateVariable, min, max, csv, color = 1):
        """Select a group of lines based on theri end state variable value
        INPUTS: the name, min and max H, and the color for this group's plots
        OUTPUT: selection as class instance"""
        if self.selection is not None:
            print('Group selection already created')
            return None
        self.selection = csv.loc[(pd.Series(csv[stateVariable], dtype=float) >= min) & (pd.Series(csv[stateVariable], dtype=float) <= max)]
        self.selection.set_index(self.selection['seeds'], drop = False, append = False, inplace = True)
        self.original = self.selection.copy()
        print(f'{self.selection.shape[0]} runs with {stateVariable} between {min} and {max}')

    def createSelectionFromCSV_seeds(self, stateVariable,seeds, color=1):
        return None

    def set_OBS_PERIOD(self,value):
        self.OBS_PERIOD = value

    def get_seeds(self):
        """Returns the list of seeds of the respective selection"""
        return(self.selection["seeds"].to_list())

    def get_Lines(self):
        """Return all the lines in the selection with its params"""
        return self.selection

    def get_paramsFromSeed(self, seed):
        return self.selection.loc[seed]

    def retrieve_from_disk(self, stateVariables, seed = False):
        """Retrives the data  from the disk
        INPUT: stateVariables is a list of str with the var names,
        OUTPUT: a dictionary of data frames"""
        if seed:
            seeds = [seed]
        else:
            seeds = self.get_seeds()
        simdata = {}
        for seed in seeds:
            if seed not in simdata:
                simdata[seed] = {}
            for stateVariable in stateVariables:
                for entry in os.scandir(f"{self.working_directory}/{self.selectionName}"):
                    if ((str(seed) in entry.name.split("_")) and
                            (str(stateVariable) in entry.name.split("_")) and
                            (str(f'OBS{self.OBS_PERIOD}') in entry.name.split("_")) and
                            (str(f'pi{(self.policy).replace("_","")}') in entry.name.split("_")) and
                            (str(f'var{self.varsigma}') in entry.name.split("_"))):
                        simdata[seed].update( {stateVariable: pd.read_csv(entry.path)})
        return (simdata)

    def retrieve_run_model(self, stateVariables, seed = False, save = True):
        """Run the model for the seeds in selectionName, observing stateVariables with OBS_PERIOD
        INPUT: stateVariables is a list of str with the var names, OBS_PERIOD is an int, selectionName a str
        OUTPUT: dictionary of seed: {stateVariables: results}"""
        if seed:
            seeds = [seed]
        else:
            seeds = self.get_seeds()

        simdata = {}
        for seed in seeds:
            if self.selection is None:
                print('No selection created yet')
                return None
            lineParams = self.get_paramsFromSeed(seed)
            params = lineParams.to_dict()
            params['seed'] = params.pop('seeds')
            params['PROVIDER_INIT'] = ['applyFixed']
            params['PATIENT_INIT'] = ['applyFixed']
            params["OBS_PERIOD"] = [self.OBS_PERIOD]
            params["reproduce_line"]= ["true"]
            params['Pi'] = [self.policy]
            if 'H' in params:
                params.pop('H')

            for stateVariable in stateVariables:
                params[f"obs{stateVariable}"] = "true"
                print(f"(lineGroup) retreiving line {seed} for {stateVariable}")
                run_data = self.c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(params))
                self.c.start_server()
                simdata[seed] = {stateVariable:pd.DataFrame(run_data[0][stateVariable], columns=run_data[0]['windows'])}

        if save:
            self.save_run_to_disk(simdata = simdata)
        return (simdata)

    def save_run_to_disk(self, simdata):
        """Saves run to disk with:
        simulation file identifier: f"var{varsigma}_pi{policy}_{H}_{SEED}_OBS{OBSPERIOD}"
        """
        for seed in simdata:
            for stateVariable in simdata[seed]:
                simdata[seed][stateVariable].to_csv(
                    f"{self.working_directory}/{self.selectionName}/var{self.varsigma}_pi{(self.policy).replace("_","")}_{stateVariable}_{seed}_OBS{self.OBS_PERIOD}"
                                     , index=False)

    def produce(self, stateVariables, seeds = False):
        """For each necesary run results, retrieve them from disk, or run them from engine if not available"""

        if not seeds:
            seeds = self.get_seeds()
        results = {}
        results_seed={}
        for seed in seeds:
            results_seed = self.retrieve_from_disk(stateVariables, seed)
            for stateVariable in stateVariables:
                if seed not in results_seed:
                    results_seed[seed] = {}
                if stateVariable not in results_seed[seed]:
                    results_seed[seed].update(self.retrieve_run_model(stateVariables = [stateVariable], seed = seed)[seed])
            results.update(results_seed)
        return results

    def tweakLine(self, seed, params):
        for key, value in params.items():
            self.selection.loc[(seed), key] = value

    def untweakLine(self, seed):
        self.selection.loc[seed] = self.original.loc[seed]

    def find_5_traj(self, stateVar, q=None, window=200):
        """Find the trajectories with percentile 0, .25, .50, .75, .100 in data for the given window (or the q percentiles)
        Input: the selection and the window on which to evaluate the trajectories
        Output: a dict of the percentile: [seed, value]"""

        all_lines = self.produce([stateVar])
        h_200 = {}
        #oneSeed = list(all_lines.keys())[0]
        #windowIndex = str(all_lines[oneSeed]["windows"].loc[all_lines[oneSeed]["windows"]['0'] == int(window)].index[0])
        for seed in all_lines:
            h_200[all_lines[seed][stateVar][str(window)].mean()] = seed
        if q is None:
            q = [0, 25, 50, 75, 100]
        qs = np.percentile(list(h_200.keys()), q=q, axis=0, method="nearest")
        selectedSeeds = []
        for i in range(len(q)):
            selectedSeeds.append(h_200[qs[i]])
        return selectedSeeds

    def setPolicy(self, value):
        self.policy = value


