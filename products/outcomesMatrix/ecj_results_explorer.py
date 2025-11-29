from MyClasses.ecjHandler import EcjHandler
from MyClasses.linePlotter import LinePlotter
import matplotlib.pyplot as plt
import os

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'

## Will build the box-params + hist profiles for every .stat on the sub_path directory
base_java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/'
base_working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/'
#sub_path = 'agents/mono/'
sub_path = 'design/norm_kurtExp_inequality/'
#

## Parameters to print on top of each hist:
printParams =  ['Pi', 'varsigma', 'N', 'fixed_tau','totalCapacity', 'W']
#printParams = ['Pi', 'varsigma', 'fixed_psi', 'fixed_eta', 'fixed_rho', 'fixed_lambda']
par= "H" #The param for Hists

## Run
for entry in os.scandir(f"{base_java_output}/{sub_path}"):
    if os.path.isfile(entry.path) and 'stat' in entry.name:
        file = entry.name.split('.')[0]

        handler = EcjHandler(file = file, java_output=f'{base_java_output}{sub_path}',
                             working_directory=f'{base_working_directory}{sub_path}', ENGINE_PATH=ENGINE_PATH,
                             OBS_PERIOD=1)
        distanceGroup = handler.generateDistancesGroup()

        maxSeeds = distanceGroup.max_parameter(selection=file, param=par, number_of_max=13).to_list()
        p = LinePlotter()
        fig, axes = plt.subplot_mosaic([['box', 'h1', 'h2'],
                                        ['box', 'h3', 'h4'],
                                        ['h5', 'h6', 'h7'],
                                        ['h8', 'h9', 'h10'],
                                        ['h11', 'h12', 'h13']], layout='constrained', facecolor = 'ghostwhite',
                                       figsize=(10, 10))
        for seed in range(0,len(maxSeeds)):
            dd = distanceGroup.produce(stateVariables=par, selectionName=file, seeds=[maxSeeds[seed]])
            axes[f'h{seed+1}'].hist(dd[maxSeeds[seed]][par]['1'])
            axes[f'h{seed + 1}'].set_title(p.printableParams(lineParams =distanceGroup.get_seedParams(seed = maxSeeds[seed], selectionName=file),
                                                             policy=False, selectedParams = printParams), fontsize=5)
        distanceGroup.boxParameProfiles(cellNumber=0, ax=axes['box'], fig=fig)
        axes['box'].set_title(f'{sub_path} {file}')

        fig.show()
        fig.savefig(f'{base_java_output}{sub_path}/{file}.png')
