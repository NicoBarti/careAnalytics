from MyClasses.ecjHandler import EcjHandler
from MyClasses.linePlotter import LinePlotter
import matplotlib.pyplot as plt
import os
from MyClasses.pathFinderCollector import PathFinderCollector

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'

## Will build the box-params + hist profiles for every .stat on the sub_path directory
base_java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/'
base_working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/JAMPaper/'
#sub_path = 'agents/bi_objective/'
#sub_path = 'design/norm_kurtExp_inequality/'
sub_path = 'JAMPaper/'


## Parameters to print on top of each hist:
printParams =  ['Pi', 'varsigma', 'N', 'fixed_tau','totalCapacity', 'W']
#printParams = ['Pi', 'varsigma', 'fixed_psi', 'fixed_eta', 'fixed_rho', 'fixed_lambda']
par= "H" #The param for Hists
adjustedTs = {}

## Run
for entry in os.scandir(f"{base_java_output}/{sub_path}"):
    if os.path.isfile(entry.path) and 'stat' in entry.name:
        file = entry.name.split('.')[0]

        handler = EcjHandler(file = file, java_output=f'{base_java_output}{sub_path}',
                             working_directory=f'{base_working_directory}{sub_path}', ENGINE_PATH=ENGINE_PATH,
                             OBS_PERIOD=1, addParam={})

        ##Create CSV:
        if not os.path.exists(f'{base_java_output}{sub_path}{file}_allRuns.csv'):
            collector = PathFinderCollector(f'{base_java_output}{sub_path}')
            collector.buildRuns_ecj(file)
            ##in this case, the name of the stat file was the dominant policy identified, so I'll put this as the default policy
            handler.changeParam(paramName = "Pi", value = file)
            ##add the missing parameters to reproduce the lines
            handler.addParams(gridParams = {'random_delta_min': 0, 'random_delta_max': 10, "reproduce_line": True})

        distanceGroup = handler.generateDistancesGroup()
        distanceGroup.norm_N = 5000
        distanceGroup.norm_W = 50
        distanceGroup.norm_totalCapacity = 200
        distanceGroup.norm_fixed_psi = 1
        distanceGroup.norm_fixed_eta = 10
        distanceGroup.norm_fixed_rho = 10
        distanceGroup.norm_fixed_kappa = 1
        distanceGroup.norm_fixed_tau = 2
        distanceGroup.norm_fixed_lambda = 10

        adjustedTs[file] = distanceGroup.selections[file]['H'] / (
                    distanceGroup.selections[file]['N'] * distanceGroup.selections[file]['totalCapacity'])
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
print(adjustedTs)
