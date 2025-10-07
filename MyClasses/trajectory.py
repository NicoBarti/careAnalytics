import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from MyClasses.client import Client
from MyClasses.pathFinder import PathFinder
import numpy as np
import json

class Trajectory:
    """A param-identified or seed-identified line to be plotted"""

    def __init__(self,working_directory, selectionName, ENGINE_PATH,color = 1,seed = None, params = None, name = "Unnamed",
                 stateVariables = ["H"], cmap = 'viridis'):
        self.name = name
        self.c = Client(ENGINE_PATH=ENGINE_PATH)
        self.ENGINE_PATH = ENGINE_PATH
        self.c.start_server()
        self.working_directory = working_directory
        self.stateVariables = stateVariables
        self.selectionName = selectionName
        self.OBS_PERIOD = 100
        self.color = color
        self.cmap = mpl.colormaps[cmap]
        if seed is None:
            if params is not None:
                self.params = params
            else:
                print('Not enough parameters')
        else:
            self.seed = seed
            self.params = self.fetchParams()

    def changePolicy(self, policy):
        self.params["Pi"] = policy

    def setStateVarables(self,val):
        self.stateVarables = val

    def fetchParams(self):
        """Fetch the line params"""
        #toDO maybe i don't need thie method, easyer to instantiate a pathfinder and use its method
        found = False
        for entry in os.scandir(f"{self.working_directory}/{self.selectionName}"):
            if entry.is_file() and "params" in entry.name.split("_") and str(self.seed) in entry.name.split("_"):
                return pd.read_csv(entry.path)
                found = True
                break
        if not found:
            print('No params file found. Trying PathFinder methods.')
            pathFinder = PathFinder(working_directory = self.working_directory, allRuns_csv = None,
                                    ENGINE_PATH = self.ENGINE_PATH)
            return pathFinder.get_seedParams(seed = self.seed, OBS_PERIOD = self.OBS_PERIOD, selectionName = self.selectionName, filterParams = False)

    def runLine(self, fromSeed = False):
        """Calls the model passing the params"""
        # #TODO pass all parameters:
        # params = {"varsigma": self.params['varsigma'][0], "OBS_PERIOD": [self.OBS_PERIOD],
        #           "reproduce_line": ["t"],
        #           "fixed_kappa": self.params['fixed_kappa'][0],
        #           "fixed_delta": self.params['fixed_delta'][0],
        #           "N": self.params['N'][0],
        #           "fixed_lambda": self.params['fixed_lambda'][0],
        #           "fixed_tau": self.params['fixed_tau'][0],
        #           "fixed_eta": self.params['fixed_eta'][0],
        #           "W": self.params['W'][0],
        #           "fixed_capE": self.params['fixed_capE'][0],
        #           "fixed_capN": self.params['fixed_capN'][0],
        #           "fixed_psi": self.params['fixed_psi'][0],
        #           "fixed_rho": self.params['fixed_rho'][0],
        #           "totalCapacity": self.params['totalCapacity'][0],
        #           "Pi": self.params['Pi'][0],
        #           #"pathfinder": ['t']
        #           }
        # for stateVariable in self.stateVariables:
        #     params[f"obs{stateVariable}"] = "t"
        # if fromSeed:
        #     params["seed"] = self.seed
        # run_data = self.c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(params))
        # self.c.start_server()
        return run_data

    def printableParams(self, paramsPerLine = 4):
        selectedParams = ['Pi','fixed_kappa', 'fixed_delta', 'varsigma', 'N', 'fixed_lambda', 'fixed_tau', 'fixed_eta',
                          'totalCapacity', 'W',
                          'PATIENT_INIT', 'PROVIDER_INIT', 'fixed_capE', 'fixed_capN', 'fixed_psi', 'fixed_rho']
        t = self.params[selectedParams].loc[0]
        #round_t = {}
        #for name in t.index:
        #    if (type(t[name]) == float):
        #        round_t[name] = t[name].round(2)
        #    else:
        #        round_t[name] = t[name]
        #round_t = pd.Series(round_t)

        counter = 0
        internalDic = {}
        paramList = ""
        for key in selectedParams:
            internalDic[key] = t[key].round(2) if type(t[key]) != str else t[key]
            counter += 1
            if counter == paramsPerLine or key == selectedParams[-1]:
                paramList = f'{paramList} {pd.Series(internalDic).to_string().replace("\n", "  ; ").replace("      ", " ").replace("_", " ")} \n'
                internalDic = {}
                counter = 0
        #paramList = round_t.to_string().replace("\n", "  ; ").replace("      ", " ").replace("_", " ")
        return paramList

    def plotLine(self, axe, run_data, mainLine = False):
        """Plot one line with specified color and strong alpha if mainLine"""
        if mainLine:
            alpha,linewidth, linestyle, col = 1,2, 'dotted','black'
        else:
            alpha, linewidth, linestyle, col = 1,1, 'solid',self.cmap([self.color])
        axe.plot(run_data[0]["windows"], pd.DataFrame(run_data[0][self.stateVariables[0]]).mean(0),
                 color=col, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        return axe

    def plotNeedEnabeler(self, maxH):
        #This is messi:
        #pathFinder = PathFinder(working_directory=self.working_directory, allRuns_csv=None,
        #                        ENGINE_PATH=self.ENGINE_PATH,OBS_PERIOD=10, varsigma = varsigma)
        #params = pathFinder.get_seedParams(seed=self.seed, selectionName=self.selectionName,
        #                                 filterParams=False)
        #data = pathFinder.produce(stateVariables=["SimpleE", "N", "H"],
        #                          selectionName=self.selectionName,seeds = [self.seed])
        self.stateVariables = ["SimpleE", "N", "H"]
        data = self.runLine(fromSeed = True)
        normSimpleE = np.array(data[0]['SimpleE']).mean(0)
        #normSimpleE.drop('Unnamed: 0', inplace=True, errors='ignore')
        normN = np.array(data[0]['N']).mean(0)
        #normN.drop('Unnamed: 0', inplace=True, errors='ignore')
        meanH = np.array(data[0]['H']).mean(0)
        #meanH.drop('Unnamed: 0', inplace=True, errors='ignore')
        fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, facecolor='ghostwhite', figsize = (7,8) )
        axes[0].plot(data[0]['windows'], meanH, label="H", color = self.cmap([self.color]))
        axes[0].set_title(self.printableParams(), fontsize=10)
        axes[0].set_ylabel('Health status')
        if maxH is not None:
            axes[0].set_yticks(ticks=np.round(np.linspace(start=0, stop=maxH, num=3), 1))
            axes[0].set_ylim(bottom=0, top=(maxH + 0.05 * maxH))

        axes[1].plot(data[0]['windows'], normSimpleE, label = "E", color = 'green')
        axes[1].set_yticks(ticks= np.round(np.linspace(start=0, stop=float(self.params['fixed_capE']), num = 3),1))
        axes[1].set_ylim(bottom=0, top=(float(self.params['fixed_capE']) + 0.05* float(self.params['fixed_capE'])))
        axes[1].set_ylabel("Enabelers")

        axes[2].plot(data[0]['windows'], normN, label = "N", color = 'red')
        axes[2].set_yticks(ticks= np.round(np.linspace(start=0, stop=float(self.params['fixed_capN']), num = 3),1))
        axes[2].set_ylim(bottom=0, top=(float(self.params['fixed_capN']) +0.05*float(self.params['fixed_capN'])))
        axes[2].set_ylabel("Perceived needs")

        fig.suptitle(f'Line percentile {self.name} seed {self.seed}')
        return fig

    def reproduceLine(self, repetitions):
        """Re-run and plot the line (stateVariable-line) repetitions times"""
        fig, axe = plt.subplots(facecolor='ghostwhite', constrained_layout=True)
        self.plotLine(axe=axe, run_data=self.runLine(fromSeed=True, OBS_PERIOD = self.OBS_PERIOD), mainLine=True)
        for repetition in range(repetitions):
            self.plotLine(axe=axe, run_data=self.runLine(fromSeed=False, OBS_PERIOD=self.OBS_PERIOD), mainLine=False)
        fig.suptitle(f"Line {self.name} reproduced {repetitions} times with random seeds.")
        axe.set_xlabel("Time-steps", fontsize=15)
        axe.set_ylabel("H",fontsize=15)
        axe.set_title(self.printableParams(), fontsize=10)
        return(fig)

