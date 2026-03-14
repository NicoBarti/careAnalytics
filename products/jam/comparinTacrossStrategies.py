from MyClasses.linePlotter import LinePlotter
import matplotlib.pyplot as plt
import os
from MyClasses.pathFinder import PathFinder

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'

## Will build the box-params + hist profiles for every .stat on the sub_path directory
base_java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/'
#sub_path = 'agents/bi_objective/'
#sub_path = 'design/norm_kurtExp_inequality/'
sub_path = 'JAMPaper/'
base_working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/'+sub_path


def computeComparisons(strategy: str, percentile: float):
    """Compute simulations for best percentile of mainStrategy and their coparisons (the other three strategies)"""
    paths = {}
    strategies = [a for a in ['basal', 'risk', 'risk_need', 'need'] if a != strategy]
    paths[strategy] = PathFinder(working_directory=base_working_directory, allRuns_csv= f'{base_java_output+sub_path+strategy}_allRuns.csv',
                        ENGINE_PATH=ENGINE_PATH, varsigma=200,OBS_PERIOD=1,)

    paths[strategy].createSelection(name = strategy, minH = paths[strategy].allRuns_csv['Fitness'].quantile(percentile),
                                    maxH = paths[strategy].allRuns_csv['Fitness'].max(), slectionVar='Fitness')
    paths[strategy].produce(stateVariables=['H'], selectionName=strategy)
    for caompara_strat in strategies:
        csv_newPolicy = paths[strategy].selections[strategy].copy(deep=True)
        csv_newPolicy['Pi'] = caompara_strat
        paths[f'{strategy}/{caompara_strat}'] = PathFinder(working_directory=base_working_directory, allRuns_csv= csv_newPolicy,
                                           ENGINE_PATH=ENGINE_PATH, varsigma=200,OBS_PERIOD=1)

        paths[f'{strategy}/{caompara_strat}'].createSelection(name = f'{strategy}/{caompara_strat}',
                                              minH = paths[f'{strategy}/{caompara_strat}'].allRuns_csv['Fitness'].min(),
                                              maxH = paths[f'{strategy}/{caompara_strat}'].allRuns_csv['Fitness'].max(), slectionVar='Fitness')
        paths[f'{strategy}/{caompara_strat}'].produce(stateVariables=['H'], selectionName=f'{strategy}/{caompara_strat}')
    return paths


def plotComparisons(paths: dict, strategy: str, stateVar = 'H'):
    """Plot the comparisons of the mainStrategy and the other three strategies"""
    colors = {'basal': 'black', 'risk': 'red', 'risk_need': 'orange', 'need': 'green'}
    #Pick a seed from the main strategy
    for seed in paths[strategy].get_seeds(strategy):
        fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
        #plot the the results for all strategies
        strats = [strategy+'/'+a for a in ['basal', 'risk', 'risk_need', 'need'] if a != strategy]
        strats.append(strategy)
        for strat in strats:
            data = paths[strat].produce(stateVariables=[stateVar], selectionName=strat,
                                seeds=[seed])
            label = strat.removeprefix(f'{strategy}/') if f'{strategy}/' in strat else strat
            axe.plot(data[seed]["windows"]['0'], data[seed][stateVar].mean(0), color=colors[label], alpha=1,
                 label=label)
        fig.suptitle(f"Evolution of Mean H with {strategy.capitalize()} Dominance", fontsize=20)
        axe.set_xlabel("Time-steps", fontsize=15)
        axe.set_ylabel(f"Mean {stateVar}", fontsize=15)
        fig.legend()
        fig.show()


for strat in ['basal', 'risk', 'risk_need', 'need']:
    paths = computeComparisons(strat, 0.995)
    plotComparisons(paths, strat)

#The syntaxis for saving the