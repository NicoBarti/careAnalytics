from cProfile import label

import pandas as pd
import numpy as np
from scipy.stats import binom
from MyClasses.metrics import *
from MyClasses.client import *
from MyClasses.errors import *
from scripts.pairPlots import *
import matplotlib.pyplot as plt
from MyClasses.plotting import *


def do_mechanismNeedsToAttempts2(errors, gridParameters, list, N, name, model, type = 'needs'):
    """Plot the relation of needs per simulaitons teps to attepts"""
    nrow = gridParameters.loc[gridParameters[list['plotPar']] == list['vline']].index[0]
    figTitle =f'Detail of simulation with {list['plotPar']} = {list['vline']}'

    metricsDic = {}
    for i in range(0,N):
        metricsDic[f'metrics{i}'] = errors.dic[f'metrics{i}']

    averageN_sim = np.empty([0,150])
    averageB_sim = np.empty([0,150])
    averageE_sim = np.empty([0,150])

    for repetition in range(0, len(metricsDic)):
        H = metricsDic[f'metrics{repetition}']._fetch(name = "H", paramRowNumber=nrow, par1='capacity', par2='weeks')[0]
        averageN_sim = np.append(averageN_sim, [H.loc[:,1:].sum(0)], axis=0)
        B = metricsDic[f'metrics{repetition}']._fetch(name = "B", paramRowNumber=nrow, par1='capacity', par2='weeks')[0]
        averageB_sim = np.append(averageB_sim, [B.loc[:,1:].sum(0)], axis=0)
        E = metricsDic[f'metrics{repetition}']._fetch(name = "exp", paramRowNumber=nrow, par1='capacity', par2='weeks')[0]
        averageE_sim = np.append(averageE_sim, [E.loc[:,1:].sum(0)], axis=0)

    if type == 'needs':
        average_sim = averageN_sim
    else:
        average_sim = averageE_sim

    fig, Bheight, height = plotNeedsorExp(average_sim=average_sim, averageB_sim =averageB_sim, gridParameters = gridParameters,type = type)
    fig.suptitle(figTitle)
    fig.show()
    #fig.savefig(f'./figures/{model}/participation/DISEASE_SEVERITY__SUBJECTIVE_INITIATIVE/needsPerStep_{name}.png')

    return(fig, Bheight, height)



def plotNeedsorExp(average_sim, averageB_sim, gridParameters,type = 'needs'):
    p = Plotter()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')
    p.set_colors(type)
    height = average_sim.mean(0)
    line = height.mean().round(2)
    ax.bar(np.arange(1, 151), height, color=p.get_colors(), label=type)
    ax.hlines(line, 0, 150, colors=p.get_colors(), linestyles='dashed')
    Bheight = averageB_sim.mean(0)
    Bline = Bheight.mean().round(2)
    ax.bar(np.arange(1, 151), Bheight, color='black', label='Attempts')
    ax.hlines(Bline, 0, 150, colors='black', linestyles='dashed')
    maxAttPatient = gridParameters['numPatients'][0]
    ax.hlines(maxAttPatient, 0, 150, colors='black', linestyles='solid')
    maxNeedPatient = gridParameters['numPatients'][0] * 5
    ax.hlines(maxNeedPatient, 0, 150, colors=p.get_colors(), linestyles='solid')

    ax.set_ylim(0, maxNeedPatient)
    newTicks = np.append(ax.get_yticks(), np.array([line, Bline, maxAttPatient, maxNeedPatient]))
    newLabs = np.round(np.append(ax.get_yticks(), np.array([line, Bline])), 1).tolist()
    newLabs.append('max B')
    if type == 'needs':
        newLabs.append('max N')
        ax.set_ylabel('Needs and attempts per step')

    else:
        newLabs.append('max E')
        ax.set_ylabel('Expectations and attempts per step')

    ax.set_yticks(ticks=newTicks, labels=newLabs)
    ax.set_xlabel('Time steps')
    ax.legend()

    return fig, Bheight.mean(), height.mean()

#    fig.suptitle(figTitle)
 #   fig.show()
  #  fig.savefig(f'./figures/{model}/participation/DISEASE_SEVERITY__SUBJECTIVE_INITIATIVE/needsPerStep_{name}.png')