import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from MyClasses.client import Client
from MyClasses.pathFinder import PathFinder
import numpy as np
import json

class LinePlotter:

    def __init__(self, cmap = 'viridis', color = 1):
        self.cmap = mpl.colormaps[cmap]
        self.color = 1
        self.selectedParams = ['Pi','fixed_kappa', 'fixed_delta', 'varsigma', 'N', 'fixed_lambda', 'fixed_tau', 'fixed_eta',
                          'totalCapacity', 'W',
                          'PATIENT_INIT', 'PROVIDER_INIT', 'fixed_capE', 'fixed_capN', 'fixed_psi', 'fixed_rho', 'seeds']

    def setColor(self,value):
        self.color = value

    def setColorMAP(self, value):
        self.cmap = mpl.colormaps[value]

    def printableParams(self, lineParams, paramsPerLine = 4, policy = True, selectedParams = None):
        """Print params in readeable lines """

        if selectedParams is None:
            selectedParams = self.selectedParams

        counter = 0
        internalDic = {}
        paramList = ""
        thisParams = selectedParams if policy else [x for x in selectedParams if x != "Pi"]
        thisParams = [x for x in selectedParams if x in lineParams] #Avoids failing to params not included
        for key in thisParams:
            internalDic[key] = lineParams[key].round(2) if type(lineParams[key]) != str else lineParams[key]
            counter += 1
            if counter == paramsPerLine or key == thisParams[-1]:
                s_str = pd.Series(internalDic, dtype='str').to_string().replace('\n', '  ; ').replace('      ', ' ').replace('_', ' ')
                paramList = f"{paramList} {s_str} \n"
                internalDic = {}
                counter = 0
        return paramList

    def plotLine(self, axe, run_data, stateVariable,mainLine = False, label = None):
        """Plot one line with specified color and strong alpha if mainLine"""
        if mainLine:
            alpha,linewidth, linestyle, col = 1,2, 'dotted','black'
        else:
            alpha, linewidth, linestyle, col = 1,1, 'solid',self.cmap([self.color])
        axe.plot(run_data[stateVariable].mean(0),
                 color=col, alpha=alpha, linewidth=linewidth, linestyle=linestyle,
                 label=label)
        axe.set_xlabel("Time")
        axe.set_ylabel(stateVariable)
        return axe

    def hist(self, data, stateVar, axe, window, x_texsize = 20):
        """Plot histogram. If it T, remove zeros"""

        #Extract window
        try:
            dataToPlot = data[stateVar][str(window)]
        except:
            dataToPlot = data[stateVar][window]

        #If T, remove 0s
        if stateVar == "T":
            dataToPlot = dataToPlot[dataToPlot != 0]

        axe.hist(dataToPlot)
        axe.set_xlabel(stateVar, size=x_texsize)
        axe.set_ylabel("Count")
        return axe

    def same_limits_y(self, axes , fixMin = None, fixMax = None):
        min, max = [], []
        for axe in axes:
            max.append(axe.get_ylim()[1])
            min.append(axe.get_ylim()[0])
        newMAXLim = np.max(np.array(max)) if fixMax is None else fixMax
        newMINLim = np.min(np.array(min)) if fixMin is None else fixMin
        for axe in axes:
            axe.set_ylim(bottom =newMINLim, top= newMAXLim )
        return axes

    def same_limits_x(self, axes,fixMin = None, fixMax = None):
        min,max = [],[]
        for axe in axes:
            max.append(axe.get_xlim()[1])
            min.append(axe.get_xlim()[0])
        newMAXLim = np.max(np.array(max)) if fixMax is None else fixMax
        newMINLim = np.min(np.array(min)) if fixMin is None else fixMin
        for axe in axes:
            axe.set_xlim(left = newMINLim, right = newMAXLim)
        return axes

    #def individual_trayectories(self, axe, stateVar):






