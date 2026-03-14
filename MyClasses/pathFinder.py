from typing import Any

import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from MyClasses.client import Client
import numpy as np
import json

#from careEngine6.drafts.paper_draft_Script import selectionName


class PathFinder:
    """Make inspection grids"""
    #ToDO this class shouls only retreive models and find paths. The inspection (grids) should be another class that inherits

## Files names:
## each directory inside the working directory contains one selection, and is named by the name of the selection
## each simulation file has stateVariable_seed_OBSPERIOD

    def __init__(self, working_directory, allRuns_csv, ENGINE_PATH,varsigma,OBS_PERIOD, filterW_capacity = False,
                 oldstyle = False, indexVar = 'H'):
        ##allRuns_csv is a direct output from Java with the end values of H (culmn data) and the seed for each run.
        self.working_directory = working_directory
        self.varsigma = varsigma
        self.indexVar = indexVar
        if type(allRuns_csv) == str:
            self.allRuns_csv = pd.read_csv(allRuns_csv)
            self.allRuns_csv.rename(columns={indexVar:'data'}, inplace=True)
        else:
            print('Using provided .csv file. If errors, try passing the path and I handle types and names')
            self.allRuns_csv = allRuns_csv
        self.ENGINE_PATH = ENGINE_PATH
        self.selections = {}
        self.cmap  = mpl.colormaps['viridis']
        self.colors = {}
        print("recomendation: make a dir varsigmaNamed as working_directory")
        self.c = Client(ENGINE_PATH=ENGINE_PATH)
        self.c.start_server()
        self.OBS_PERIOD = OBS_PERIOD
        self.filterW_capacity = filterW_capacity
        self.oldstyle = oldstyle

    def makedir(self, selectionName):
        """Create a directory for every selection"""
        if not os.path.exists(self.working_directory + "/" + selectionName):
            os.makedirs(self.working_directory + "/" + selectionName)

    def createSelection(self, name, minH, maxH, color = 1, slectionVar = 'H'):
        """Select a group of lines based on theri end H value
        INPUTS: the name, min and max H, and the color for this group's plots
        OUTPUT: selection as class instance"""
        selection = self.allRuns_csv.loc[(self.allRuns_csv[slectionVar] >= minH) & (self.allRuns_csv[slectionVar] <= maxH)]
        self.selections[name] = selection
        self.makedir(name)
        print(f'{selection.shape[0]} runs with {slectionVar} between {minH} and {maxH}')
        if self.filterW_capacity:
            print("(PathFinder) Filtering totalCapacity/W < 200")
            indexFilter = []
            for index, data in selection.iterrows():
                params = self.get_seedParams(seed = data["seeds"], selectionName=name)
                if float(params["totalCapacity"])/float(params["W"]) < 200:
                    indexFilter.append(index)
            selection = selection.filter(items = indexFilter, axis = 0)
        print(f"Created selection {name} with {selection.shape[0]} runs")

        self.colors[name] = color

    def set_OBS_PERIOD(self,value):
        self.OBS_PERIOD = value

    def get_seeds(self, selection):
        """Returns the list of seeds of the respective selection"""
        return(self.selections[selection]["seeds"].to_list())

    def retrieve_from_disk(self, stateVariables, selectionName, seed = False):
        """Retrives the data,  from the disk
        INPUT: stateVariables is a list of str with the var names, OBS_PERIOD is an int, selectionName a str
        OUTPUT: a dictionary of data frames"""
        ## simulation file identifier: stateVariable_seed_OBSPERIOD
        if seed:
            seeds = [seed]
        else:
            seeds = self.get_seeds(selectionName)
        simdata = {}
        for seed in seeds:
            for stateVariable in stateVariables:
                for entry in os.scandir(f"{self.working_directory}/{selectionName}"):
                    if (str(seed) in entry.name.split("_")) and (str(stateVariable) in entry.name.split("_")) and (str(self.OBS_PERIOD) in entry.name.split("_")):
                        simdata[seed] =  {stateVariable: pd.read_csv(entry.path)}
        return (simdata)

    def retrieve_run_model(self, stateVariables, selectionName, seed = False, tweak = {}):
        """Run the model for the seeds in selectionName, observing stateVariables with OBS_PERIOD
        INPUT: stateVariables is a list of str with the var names, OBS_PERIOD is an int, selectionName a str
        OUTPUT: dictionary of seed: {stateVariables: results}"""
        if seed:
            seeds = [seed]
        else:
            seeds = self.get_seeds(selectionName)
        if stateVariables[0] == "params": #handle the case where params were not retreived, re-runs with H
            stateVariables = [self.indexVar]
        simdata = {}
        for seed in seeds:
            if self.oldstyle:
                params = {"varsigma": [self.varsigma], "OBS_PERIOD": [int(self.OBS_PERIOD)], "seed": [int(seed)],
                      "pathfinder":["t"]}
            else:
                lineParams = self.selections[selectionName][self.selections[selectionName]['seeds']==seed]

                params = lineParams.to_dict(orient="list")
                params['seed'] = params.pop('seeds')
                params['PROVIDER_INIT'] = ['applyFixed']
                params['PATIENT_INIT'] = ['applyFixed']
                params["OBS_PERIOD"] = [self.OBS_PERIOD]

                for key, value in tweak.items():
                    params[key] = value

                if 'H' in params:
                    params.pop('H')
                if 'Fitness' in params:
                    params.pop('Fitness')


            for stateVariable in stateVariables:
                params[f"obs{stateVariable}"] = "true"
            print(f"(PathFinder) retreiving seed {seed}")
            run_data = self.c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(params))
            #simdata[seed] = {stateVariable: pd.DataFrame(run_data[0][stateVariable], columns = [str(x) for x in run_data[0]["windows"]]),
            #                 "windows": pd.DataFrame({'0':run_data[0]["windows"]})}
            receivedParams = json.loads(bytes(self.c.received_params.strip()))
            #handle multidimentional outputs
            #TODO parametrize if min, max, mean ,etc
            if stateVariable in ['E']:
                run_data = self.filterMaxValuePerProvider(run_data = run_data, stateVariable = stateVariable)

            win = pd.DataFrame({'0':run_data[0]["windows"]})
            simdata[seed] = {stateVariable: pd.DataFrame(run_data[0][stateVariable], columns = [str(x) for x in pd.DataFrame({'0':run_data[0]["windows"]}).index.to_list()]),
                             "windows": win,
                             "params": pd.DataFrame(receivedParams, index=[0])}
            self.c.start_server()
        if len(tweak)==0:
            self.save_run_to_disk(simdata = simdata, selectionName=selectionName)
        return (simdata)

    def filterMaxValuePerProvider(self, run_data, stateVariable):
        """Take the max value across providers at each window to per patient"""
        run_data[0][stateVariable] = np.array(run_data[0][stateVariable]).max(axis=1)
        return run_data

    def save_run_to_disk(self, simdata,selectionName):
        """Saves run to disk with:
        simulation file identifier: stateVariable_seed_OBSPERIOD
        """
        for seed in simdata:
            for stateVariable in simdata[seed]:
                simdata[seed][stateVariable].to_csv(f"{self.working_directory}/{selectionName}/{stateVariable}_{seed}_{self.OBS_PERIOD}", index=False)

    def produce(self, stateVariables: object, selectionName: object, seeds: object = [],
                tweak: object = {}) -> dict[Any, Any]:
        """For each necesary run results, retrieve them from disk, or run them from engine if not available"""
        #Todo when reading from disk, data frames contain the 'Unnamed: 0' column/(index?). When retreiving from simulation they don't.
        if len(seeds) ==0:
            seeds = self.get_seeds(selectionName)
        results = {}
        for seed in seeds:
            for stateVariable in stateVariables:
                if len(tweak) > 0:
                    simdata = self.retrieve_run_model(stateVariables=[stateVariable], selectionName=selectionName,
                                                      seed=seed, tweak = tweak)
                else:
                    existingRun = False
                    for entry in os.scandir(f"{self.working_directory}/{selectionName}"):
                        if entry.is_file() and stateVariable in entry.name.split("_") and str(seed) in entry.name.split("_") and str(self.OBS_PERIOD) in entry.name.split("_"):
                            existingRun = True
                            break
                    if existingRun:
                        simdata = self.retrieve_from_disk(stateVariables = [stateVariable], selectionName=selectionName, seed=seed)
                    else:
                       simdata = self.retrieve_run_model(stateVariables = [stateVariable],  selectionName=selectionName, seed = seed)

                if seed in results:
                    results[seed].update(simdata[seed])
                else:
                    results.update(simdata)

                if "windows" not in results[seed]:
                    win = self.retrieve_from_disk(stateVariables = ["windows"], selectionName=selectionName, seed=seed)
                    results[seed].update(win[seed])
        return results

    def checkSelection(self, selection):
        """Check if selection matches final H"""

        savedResults = self.selections[selection]
        for savedRow in range(savedResults.shape[0]):
            seed = savedResults.iloc[savedRow]["seeds"]
            results = self.produce(stateVariables = [self.indexVar], selectionName=selection, seeds = [seed])
            finalMean = np.array(results[seed][self.indexVar])[:,-1].mean()
            print(f"Saved final H: {savedResults.iloc[savedRow]["data"]}. Produced final H: {finalMean}")

    def find_5_traj(self, selection,stateVar, q=None, window=200):
        """Find the trajectories with percentile 0, .25, .50, .75, .100 in data for the given window
        Input: the selection and the window on which to evaluate the trajectories
        Output: a dict of the percentile: [seed, value]"""

        all_lines = self.produce([stateVar], selectionName=selection)
        h_200 = {}
        oneSeed = list(all_lines.keys())[0]
        windowIndex = str(all_lines[oneSeed]["windows"].loc[all_lines[oneSeed]["windows"]['0'] == int(window)].index[0])
        for seed in all_lines:
            h_200[all_lines[seed][stateVar].loc[:,windowIndex].mean()] = seed
        if q is None:
            q = [0, 25, 50, 75, 100]
        qs = np.percentile(list(h_200.keys()), q=q, axis=0, method="nearest")
        selectedSeeds = {}
        for i in range(len(q)):
            selectedSeeds[q[i]] = [h_200[qs[i]], qs[i]]
        return selectedSeeds

    def find_parameters_above_percentile(self, stateVar, selection, percentile: float = 0.8 ) -> list:
        """Find the parametes in a selection that are above the given percentile.
        To be used to find the params with best fitness"""

        qs = np.percentile(self.selections[selection][stateVar], q=percentile, axis=0, method="nearest")
        return self.selections[selection].loc[self.selections[selection][stateVar] > qs]

    def max_parameter(self, param, selection, number_of_max = 1):
        """Find the parametrization with max value"""
        #return self.selections[selection].loc[self.selections[selection][param] ==
        #                                      self.selections[selection][param].max()]['seeds'].iloc[0]
        return self.selections[selection].sort_values(by=self.indexVar, ascending=False)['seeds'].iloc[0:number_of_max]

    def plot_some_lines(self, selection, window, q = None, stateVar='H', lines = None, color = "selection"):
        """Plot the lines for the percentiles 0, 0.25, 0.5, 0.75, 1 at time window window (or q lines if given)
        Input: data, the dictionary with all the data for each type.
        type: (str) the data type
        Output: (fig, axe)"""

        if lines is None:
            percentile_seeds = self.find_5_traj(selection=selection, window=window, stateVar=stateVar, q = q)
            seeds = list(np.array(list(percentile_seeds.values()))[:,0])
            data = self.produce(stateVariables=[stateVar], selectionName=selection, seeds=seeds)
        else:
            percentile_seeds = None
            seeds = list(lines.keys())
            data = lines
        colors=[]
        if color == "selection":
            colors = [self.cmap(self.colors[selection])] * len(seeds)
        else:
            colors = mpl.colormaps[color](np.linspace(0, 1, len(seeds)))

        fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
        for i in range(len(seeds)):
            d = data[seeds[i]][stateVar].drop('Unnamed: 0', axis=1, errors = 'ignore') #in case Unamed: 0 still there
            axe.plot( data[seeds[i]]["windows"]['0'],d.mean(0),
                      color = colors[i], alpha=1)
        fig.suptitle(f"Selection of trajectories with {stateVar} ending around {selection}", fontsize=20)
        axe.set_xlabel("Time-steps", fontsize=15)
        axe.set_ylabel(f"{stateVar}",fontsize=15)
        return fig, axe, percentile_seeds

    def grid_row(self, selectionName, stateVar, seed, axes, windows):
        """Plot a row in the inspection grid
        INPUT: axes for one row, an array of 6 axes
        OUTPUT: the plotted axes"""

        data = self.produce(stateVariables=[stateVar], selectionName=selectionName, seeds=[seed])
        xmins, xmaxs, ymins, ymaxs  = [],[],[],[]
        for i in range(0,len(windows)):
            windowIndex = str(data[seed]["windows"].loc[data[seed]["windows"]['0'] == int(windows[i])].index[0])
            axes[i].hist(data[seed][stateVar].loc[:,windowIndex], density = False)
            axes[i].set_xlabel(stateVar)
        for i in range(0,len(windows)):
            xmins.append(axes[i].get_xlim()[0])
            xmaxs.append(axes[i].get_xlim()[1])
            ymins.append(axes[i].get_ylim()[0])
            ymaxs.append(axes[i].get_ylim()[1])
        xmin, xmax, ymin, ymax = min(xmins), max(xmaxs), min(ymins), max(ymaxs)
        for i in range(0, len(windows)):
            axes[i].set_xlim(xmin, xmax)
            axes[i].set_ylim(ymin, ymax)
        return(axes)

    def full_gird(self, selectionName, seed, title):
        stateVariables = ["H", "N", "SimpleE", "SimpleB","SimpleC", "T"]
        windows = [100, 200, 300, 400, 500]
        fig, axes = plt.subplots(len(stateVariables), len(windows), figsize=(6*len(windows), 6*len(stateVariables)), facecolor='ghostwhite',
                                 constrained_layout=True)
        fig.suptitle(title, fontsize=50)
        for i in range(len(stateVariables)):
            self.grid_row(selectionName=selectionName, stateVar=stateVariables[i], seed=seed, axes=axes[i], windows=windows)
        for i in range(len(windows)):
            axes[0,i].set_title(f'Step {windows[i]}')
        t= self.get_seedParams(seed = seed, selectionName=selectionName, filterParams=False)
        round_t = {}
        for name in t.index:
            if(type(t[name])==float):
                round_t[name] = t[name].round(2)
            else:
                round_t[name] = t[name]
        round_t = pd.Series(round_t)
        paramList = round_t.to_string().replace("\n", "  ; ").replace("      ", " ").replace("_", " ")
        fig.supxlabel(paramList, fontsize=13)
        print(f"{title} {paramList}")
        return(fig, t)

    def get_seedParams(self, seed, selectionName, filterParams = True):
        """Retreive the params for a seed"""
        simdata = self.produce(["params"], selectionName, seeds = [seed])
        selectedParams = ['fixed_kappa','fixed_delta','varsigma','N','fixed_lambda','fixed_tau','fixed_eta','totalCapacity','W',
                          'PATIENT_INIT', 'PROVIDER_INIT','fixed_capE','fixed_capN','fixed_psi','fixed_rho']
        if filterParams:
            return simdata[seed]["params"][selectedParams].loc[0]
        else:
            return simdata[seed]["params"].loc[0]

    def build_allRuns(self, dataPath, csvPath):
        """Read all the H and seed of the engine runs and compile a csv"""

        if os.path.exists(csvPath):
            print("Output file already exists. Will not overwrite it.")
            return

        runs = {}
        for entry in os.scandir(f"{dataPath}/H"):
            if entry.is_file() and "csv" in entry.name.split("."):
                id = entry.name.split("_")[2].split(".")[0]
                iteration = entry.name.split("_")[1].split("pathFinder")[0]
                if id not in runs:
                    runs[id] = {iteration : {self.indexVar:pd.read_csv(entry.path)}}
                else:
                    runs[id].update({iteration : {self.indexVar:pd.read_csv(entry.path)}})
        for entry in os.scandir(f"{dataPath}/seeds"):
            if entry.is_file() and "csv" in entry.name.split("."):
                id = entry.name.split("_")[2].split(".")[0]
                iteration = entry.name.split("_")[1].split("pathFinder")[0]
                runs[id][iteration].update({'seeds':pd.read_csv(entry.path)})

        Hs = np.empty(0)
        seeds = np.empty(0)
        for ids in runs:
            for iteration in runs[ids]:
                Hs = np.append(Hs, runs[ids][iteration][self.indexVar])
                seeds = np.append(seeds, runs[ids][iteration]['seeds'])

        readed = pd.DataFrame({'seeds': seeds, 'data': Hs})
        readed.to_csv(csvPath, index=False)




# pathFinderInstance = PathFinder(
#     working_directory= f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500" ,
#     ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar",
#     allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/provisionalAllRuns.csv"),
#     varsigma=500)
# #
# pathFinderInstance.build_allRuns(dataPath='/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder',
#                                  csvPath='/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/runs28_8_25_1.csv')