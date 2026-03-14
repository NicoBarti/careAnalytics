from scipy.stats import alpha

from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.pathFinder import PathFinder

from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.client import Client
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# Import the reusable plotting functions from the new module
from products.outcomesMatrix.complex_plotter import complexAxeDict, populate_axe, onePlot

p = LinePlotter()
def generateDistancesGroup(file: str, java_output: str, working_directory: str, collect = False,OBS_PERIOD=100 ) -> TrajectoryDistances:
    if collect:
        collector = PathFinderCollector(java_output)
        collector.buildRuns_ecj(file)
    data = pd.read_csv(java_output + f'{file}_allRuns.csv')
    data.rename(columns={'Fitness': "H"}, inplace=True)
    #Pick the last 40 generations
    data = data.loc[data.shape[0]-40:data.shape[0]]
    distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                              ENGINE_PATH=ENGINE_PATH,
                                              varsigma=100, selectionName=file, minH=0, maxH=1000,
                                              OBS_PERIOD=OBS_PERIOD, oederByWindow=100,
                                              orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3}, addParam={'Pi'})
    distances.set_grain(data.shape[0])
    return distances
def build_deciles_frame(data: pd.DataFrame) -> pd.DataFrame:
    #create the DataFrame
    output = pd.DataFrame(index = data.index)
    #Do for every time window
    for window, values in data.items():
        #Compute the deciles for this window
        deciles = np.quantile(values, np.linspace(start=0.1, stop=1, num = 10))
        #construct a Serie with the decile values for each index
        order = []
        for value in values:
            for decile in range(0,len(deciles)):
                o = 1 #defaults to last decile
                if value <= deciles[decile]:
                    o = decile+1
                    break
            order.append(o) #assign decile
        s = pd.Series(data = order, index = values.index,name=int(window))
        # Construct a dataframe with the deciles series per window
        output = output.join(s)
    return output
def cummulativeStateVariable(data: dict) -> pd.DataFrame:
    """Compute the cummulative sum of a state variable per patient and save into the dictionary"""
    return data.cumsum(axis=1)
def diffStateVariable(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the difference of a state variable per patient and save into the dictionary"""
    return data.diff(axis=1)

jamFigure, jamAxes = plt.subplots(nrows=3, ncols= 3, figsize=(40, 30), constrained_layout=True, facecolor='white',)
jamHist, jamHistAx = plt.subplots(nrows=1, ncols= 3, figsize=(15, 5), constrained_layout=True, facecolor='white',)
jamTime, jamTimeAx = plt.subplots(nrows=1, ncols= 2, figsize=(10, 5), constrained_layout=True, facecolor='white',)
#treatments = ['hiPsi/basal','hiPsi/risk','hiPsi/need','lowPsi/basal','lowPsi/risk','lowPsi/need']
treatments = ['hiPsi/basal','hiPsi/risk','hiPsi/need']

for treatment in treatments:
    ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
    java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
    #working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'
    working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/'
    varsigma = 100
    #treatment = 'lowPsi/need'
    data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/{treatment}.csv')
    selection = treatment
    distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                    ENGINE_PATH=ENGINE_PATH,
                                    varsigma=varsigma, selectionName=selection, minH=0, maxH=1000,
                                    OBS_PERIOD=1, oederByWindow=10,
                                    orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                    addParam={'Pi'})
    #stateVariables= ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise', 'InstExp']
    stateVariables= ['H', 'N', 'E', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise', 'InstExp']

    #seed = 159753458941
    #seed = 136262753023 # no noise seed
    seed = 62920828945

    #dd = distances.produce(stateVariables=stateVariables, selectionName='dynamicLine', seeds=[62920828944])[62920828944]
    # tweak={'W':1})[int(seedWithMaxFitnes.iloc[0])]
    dd = distances.produce(stateVariables=stateVariables, selectionName=selection, seeds=[seed])[seed]

    params = p.printableParams(distances.get_seedParams(seed = seed, selectionName=selection, filterParams= False),
                               paramsPerLine=8, policy=True)


    # if treatment in ['hiPsi/basal','hiPsi/risk','hiPsi/need']:
    #     if treatment == 'hiPsi/basal':
    #         jamAxes[0,0].hist(dd['H']['100'], bins=25)
    #         jamAxes[0,0].set_xlim(left = 0, right = 25)
    #         jamAxes[0,0].set_ylim(bottom=0, top=1200)
    #         jamAxes[0,0].set_xlabel("Unserved needs", fontsize=30)
    #
    #         jamAxes[0,1] = onePlot(jamAxes[0,1], dd=dd, plotVariable='SimpleB', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[0,1].set_ylim(bottom=0, top=0.4)
    #         jamAxes[0,1].set_xlabel("Care seeking attempts", fontsize=30)
    #
    #         jamAxes[0,2] = onePlot( jamAxes[0,2], dd=dd, plotVariable='T', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[0,2].set_ylim(bottom=0, top=0.08)
    #         jamAxes[0,2].set_xlabel("Delivered treatment", fontsize=30)
    #
    #     if treatment == "hiPsi/risk":
    #         jamAxes[1,0].hist(dd['H']['100'], bins=25)
    #         jamAxes[1,0].set_xlim(left = 0, right = 25)
    #         jamAxes[1,0].set_ylim(bottom=0, top=1200)
    #         jamAxes[1,0].set_xlabel("Unserved needs", fontsize=30)
    #
    #         jamAxes[1,1] = onePlot(jamAxes[1,1], dd=dd, plotVariable='SimpleB', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[1,1].set_ylim(bottom = 0, top = 0.4)
    #         jamAxes[1,1].set_xlabel("Care seeking attempts", fontsize=30)
    #
    #         jamAxes[1,2] = onePlot( jamAxes[1,2], dd=dd, plotVariable='T', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[1,2].set_ylim(bottom=0, top=0.08)
    #         jamAxes[1,2].set_xlabel("Delivered treatment", fontsize=30)
    #
    #     if treatment == 'hiPsi/need':
    #         jamAxes[2,0].hist(dd['N']['100'], bins=25)
    #         jamAxes[2,0].set_xlim(left = 0, right = 30)
    #         jamAxes[2,0].set_ylim(bottom=0, top=1200)
    #         jamAxes[2,0].set_xlabel("Unserved needs", fontsize=30)
    #
    #         jamAxes[2,1] = onePlot(jamAxes[2,1], dd=dd, plotVariable='SimpleB', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[2,1].set_ylim(bottom=0, top=0.4)
    #         jamAxes[2,1].set_xlabel("Care seeking attempts", fontsize=30)
    #
    #         jamAxes[2,2] = onePlot( jamAxes[2,2], dd=dd, plotVariable='T', color = 'blue',low_h_cut = 0, high_h_cut = 10000, plotData = None)
    #         jamAxes[2,2].set_ylim(bottom=0, top=0.08)
    #         jamAxes[2,2].set_xlabel("Delivered treatment", fontsize=30)




    ############################
    # Getting error bars

    ##Producing the lines rep times with different seeds:
    parameters_recover = distances.get_seedParams(seed=seed, selectionName=selection, filterParams=False)
    parameters_recover['seeds'] = parameters_recover['seed']
    #parameters_recover['varsigma'] = 300

    # producer.makedir(selectionName=treatment)

    reps = 100
    means = []
    ss = 52920828945
    indexOfSimulations = pd.DataFrame(columns = parameters_recover.index.to_list().append("H"))
    for i in range(reps):
        newParams = {}
        for index, value in parameters_recover.items():
            if value:  # not include false values
                newParams[index] = [value]
            if value == "True":
                newParams[index] = ["true"]
        newParams["obsH"] = True
        newParams["seeds"] = ss
        newParams['H']= 8 ##Just to create a PathFinder
        indexOfSimulations = pd.concat([indexOfSimulations, pd.DataFrame(newParams)])
        ss = ss + 1485
    producer = PathFinder(working_directory=f'{working_directory}/errorBars/', ENGINE_PATH=ENGINE_PATH,
                          varsigma=parameters_recover['varsigma'],
                          OBS_PERIOD=parameters_recover['OBS_PERIOD'], allRuns_csv=indexOfSimulations)
    producer.createSelection(name= treatment, minH = 0, maxH = 1200)
    errorData = producer.produce(selectionName=treatment, stateVariables=['H', 'T', 'SimpleB'])

    cmap = mpl.colormaps['plasma']
    ### Make the 3 H histograms with error bars
    histCounts = np.empty((0,19))
    for key, value in errorData.items():
        a = np.array(errorData[key]['H'])[:,100] # last window, all patients
        counts, bins = np.histogram(a, bins=range(20))
        histCounts = np.concatenate((histCounts, np.array([counts])), axis=0)
    if treatment == 'hiPsi/basal':
        axNum, tit, l = 0, "First-come, first-served", "dotted"
    elif treatment == 'hiPsi/risk':
        axNum, tit, l = 1, "Risk-stratified", "dashed"
    elif treatment == 'hiPsi/need':
        axNum, tit, l = 2, "Patient-reported needs", "solid"
    lowerErrors = np.quantile(histCounts, q=0.05, axis=0)
    upperErrors = np.quantile(histCounts, q=0.95, axis=0)
    x_error_pos = np.arange(start=0.5, stop=19, step=1)
    jamHistAx[axNum].stairs(histCounts.mean(0), bins, fill = True, color = cmap(0.6-(axNum*0.3)))
    jamHistAx[axNum].errorbar(x = x_error_pos, y = histCounts.mean(0), yerr = np.absolute(np.quantile(histCounts, q=[0.05,0.95], axis=0)- histCounts.mean(0)), fmt='.')
    jamHistAx[axNum].set_ylim(top=1200)
    jamHistAx[axNum].set_title(tit)
    jamHistAx[axNum].set_xlabel("Number of unserved needs per patient")
    jamHistAx[axNum].set_ylabel("Number of patients")
    jamHist.suptitle("Distribution of unserved needs by the end of the simulations", size = 15)

    ### Compara Treatments and participation
    #participation
    seekMean = np.empty((0,101))

    x_error_pos = np.arange(start=0, stop=101, step=1)

    for key, value in errorData.items():
        #a = np.array(errorData[key]['SimpleB']).mean(0) # last window, all patients
        a = np.array(errorData[key]['SimpleB']).sum(0) # last window, all patients

        seekMean = np.concatenate((seekMean, np.array([a])), axis=0)
    jamTimeAx[0].plot(range(101), seekMean.mean(0), label = tit, linestyle = l, color = cmap(0.6-(axNum*0.3)))
    jamTimeAx[0].errorbar(x=x_error_pos, y=seekMean.mean(0),
                         yerr=np.absolute(np.quantile(seekMean, q=[0.05, 0.95], axis=0) - seekMean.mean(0)), fmt=',',
                         ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2)

    jamTimeAx[0].legend()
    jamTimeAx[0].set_xlabel("Simulation time (cycles)")
    jamTimeAx[0].set_ylabel("Number of patients asking for appointments")
    jamTimeAx[0].set_title("Participation")
    #treatments
    treatMean = np.empty((0,101))
    for key, value in errorData.items():
        #a = np.array(errorData[key]['T']).mean(0) # last window, all patients
        b = np.array(errorData[key]['T'])
        np.putmask(b, b == 0, np.nan) # not consider 0 treatments
        a = np.nanmean(b, axis = 0)
        treatMean = np.concatenate((treatMean, np.array([a])), axis=0)
    jamTimeAx[1].plot(range(101), treatMean.mean(0), label = tit, linestyle = l, color = cmap(0.6-(axNum*0.3)))
    jamTimeAx[1].errorbar(x=x_error_pos, y=treatMean.mean(0),
                          yerr=np.absolute(np.quantile(treatMean, q=[0.05, 0.95], axis=0) - treatMean.mean(0)), fmt=',',
                          ecolor=cmap(0.6-(axNum * 0.3)), elinewidth=0.2)
    jamTimeAx[1].legend()
    jamTimeAx[1].set_xlabel("Simulation time (cycles)")
    jamTimeAx[1].set_ylabel("Average needs solved per patient (treatment)")
    jamTimeAx[1].set_title("Treatment")
    jamTime.suptitle("Temporal trajectories of participation and treatment", size = 15)


    # c = Client(ENGINE_PATH=ENGINE_PATH)
    # c.start_server()
    # parameters_recover = distances.get_seedParams(seed = seed, selectionName=selection, filterParams= False)
    # reps = 100
    # means = []
    # ss = 52920828945
    # np.empty((reps, parameters_recover['']))
    # for i in range(reps):
    #     newParams = {}
    #     for index, value in parameters_recover.items():
    #         if value: #not include false values
    #             newParams[index] = [value]
    #         if value == "True":
    #             newParams[index] = ["true"]
    #     newParams["obsH"] = True
    #     newParams["seed"] = ss
    #     d = c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(newParams))
    #     c.start_server()
    #     win = pd.DataFrame({'0':d[0]["windows"]})
    #     Frame_d = pd.DataFrame(d[0]["H"], columns = [str(x) for x in pd.DataFrame({'0':d[0]["windows"]}).index.to_list()])
    #     means.append(Frame_d['100'].mean())
    #     ss = ss + 1485
    # means = np.array(means)
    # print(f'{treatment} H mean: {np.array(means).mean()}')
    # print(f'{treatment} H q .05 - .95: {np.quantile(means,[.05])} - {np.quantile(means,[.95] )}')

    # hiPsi/basal H mean: 4.663918536535389
    # hiPsi/basal H q .05 - .95: [4.48797248] - [4.85827894]
    # hiPsi/risk H mean: 4.510811443651311
    # hiPsi/risk H q .05 - .95: [4.38749415] - [4.63099705]
    # hiPsi/need H mean: 3.1273967152071935
    # hiPsi/need H q .05 - .95: [2.92968665] - [3.32955182]
    #
    # lowPsi/basal H mean: 4.329819563439796
    # lowPsi/basal H q .05 - .95: [4.09185042] - [4.55959434]
    # lowPsi/risk H mean: 2.999301323065632
    # lowPsi/risk H q .05 - .95: [2.78571412] - [3.17743617]
    # lowPsi/need H mean: 2.875151206996474
    # lowPsi/need H q .05 - .95: [2.57371702] - [3.18140144]

    ############################

    ############################
    #Print separated graphs for paper figure
    # allData = np.quantile(np.array(dd['H']['100']), q=[1])[0]
    # cmap = mpl.colormaps['plasma']
    # color = cmap(0.1)
    # k = 'N'
    # fig, axes = plt.subplots(1,3, figsize=(15,5), constrained_layout=True, facecolor='ghostwhite')
    # #fig, ax = plt.subplots(1,1, figsize=(5,5), constrained_layout=True, facecolor='ghostwhite')
    # axes[0] = onePlot(axes[0], dd=dd, plotVariable=k, color=color, low_h_cut=0, high_h_cut=allData)
    # axes[0].set_xlabel('Simulation cycles', fontsize = 10)
    # axes[0].set_ylabel('Average needs per patient', fontsize = 12)
    # axes[0].set_ylim(bottom = 5)
    # k = 'SimpleB'
    # #fig, ax = plt.subplots(1,1, figsize=(5,5), constrained_layout=True, facecolor='ghostwhite')
    # axes[1] = onePlot(axes[1], dd=dd, plotVariable=k, color=color, low_h_cut=0, high_h_cut=allData)
    # axes[1].set_xlabel('Simulation cycles', fontsize = 10)
    # axes[1].set_ylabel('Average care-seeking per patient', fontsize = 12)
    # k = 'T'
    # #fig, ax = plt.subplots(1,1, figsize=(5,5), constrained_layout=True, facecolor='ghostwhite')
    # axes[2] = onePlot(axes[2], dd=dd, plotVariable=k, color=color, low_h_cut=0, high_h_cut=allData)
    # axes[2].set_xlabel('Simulation cycles', fontsize = 10)
    # axes[2].set_ylabel('Average needs solved per patient', fontsize = 12)
    # fig.show()
    #
    # c = Client(ENGINE_PATH=ENGINE_PATH)
    # thisSeedPars = distances.get_seedParams(seed = seed, selectionName=selection, filterParams= False)
    # newParams={}
    # for index, value in thisSeedPars.items():
    #     if value: #not include false values
    #         newParams[index] = [value]
    #     if value == "True":
    #         newParams[index] = ["true"]
    # newParams["reproduce_line"]= ["true"]
    # #newParams.pop("pathfinder")
    # run_basal = c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(newParams))
    #

    ### Test the effect of policy:
    # Basal policy

    # Risk policy
    ############################
    ##THIS ARE THE LINES FOR THE MAIN PLOTS:
    allData = np.quantile(np.array(dd['H'][f'{varsigma}']), q=[1])[0]
    populate_axe(complexAxeDict(), dd=dd, low_h_cut = 0, high_h_cut = allData,sub_title = params)
    print(f'Final H in the whole population: {dd["H"][f'{varsigma}'].mean()}')
    print(f'Final E in the whole population: {dd["E"][f'{varsigma}'].mean()}')
    plt.suptitle(f'All population seed: {seed} {treatment}', fontsize=50)
    plt.show()
    #print(f'Final E in the whole population: {dd["SimpleE"][f'{varsigma}'].mean()}')
    #fig, ax = plt.subplots(1,1, figsize=(5, 5), constrained_layout=True, facecolor='ghostwhite',)
    #ax.hist(dd['H']['100'], bins=50)
    #fig.suptitle(f'Health status at the end of the simulation {treatment}')
    #fig.show()


    # Coupling of needs and exp
    import seaborn as sns

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True, facecolor='ghostwhite', )
    # ax.plot(dd['N'].to_numpy().flatten(), dd['E'].to_numpy().flatten(), alpha = 0.3, color = 'blue')
    # ax.set_xlabel('N')
    # ax.set_ylabel('E')
#    print(f'correlation E and N in {treatment}: {np.corrcoef(dd['N'].to_numpy().flatten(), dd['E'].to_numpy().flatten())}')
    ############################
    #Dynamics per decile
    # lowerDecile = np.quantile(np.array(dd['H']['100']), q=[0.1])[0]
    # populate_axe(complexAxeDict(), low_h_cut = 0, high_h_cut = lowerDecile,sub_title = params)
    # plt.suptitle(f'Lower Decile (best H outcome) seed {seed}', fontsize = 50)
    # plt.show()
    #
    # upperDecile = np.quantile(np.array(dd['H']['100']), q=[0.9])[0]
    # populate_axe(complexAxeDict(),low_h_cut = upperDecile, high_h_cut = allData, sub_title = params)
    # plt.suptitle(f'Upper Decile (worst H outcome) seed: {seed} ', fontsize = 50)
    # plt.show()
    #
    # # excluded pop:
    # populate_axe(complexAxeDict(),low_h_cut = 0.000000000001, high_h_cut = allData, sub_title = params)
    # plt.suptitle(f'Excluded population seed: {seed} ', fontsize = 50)
    # plt.show()
    ############################

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
jamAxes[0,0].set_yticks([550,650], labels=["first-served ","First-come "], fontsize = 50)
jamAxes[1,0].set_yticks([600], labels=["Risk-stratified "], fontsize = 50)
jamAxes[2,0].set_yticks([650,550], labels=["Patient-reported ","needs "], fontsize = 50)


#jamAxes[0,0].set_yticks([590], labels=["first-served"], fontsize = 50)
#make_axes_area_auto_adjustable(jamAxes)
#jamFigure.show()
jamHist.show()
jamTime.show()
#stateVariables= ['InstExp']
#dd = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
#                          tweak={'W':1})[int(seedWithMaxFitnes.iloc[0])]
# plotVariables = ['H', 'N', 'N_diff', 'SimpleE', 'SimpleE_diff' ,'SimpleB','SimpleB_cum', 'SimpleC','SimpleC_cum','T','T_cum', 'Disease'
#     ,'Disease_cum', 'ExpNoise', 'ExpNoise_cum','InstExp']

# for plotVariable in plotVariables:
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), constrained_layout=True, facecolor='ghostwhite')
#     #ExtractPlotData
#     print(plotVariable)
#     plotData = dd[plotVariable.strip('_diff').strip('_cum')].loc[dd['H']['100'] <= lowerDecile]
#     #Comput diff or cum if needed
#     plotData = plotData.diff(axis=1) if '_diff' in plotVariable else plotData
#    plotData = plotData.cumsum(axis=1) if '_cum' in plotVariable else plotData
    #for index, row in plotData.iterrows():
    #    ax.plot([x for x in range(len(row))], row, color='blue', alpha=0.02)
#     ax.plot([x for x in range(plotData.shape[1])], plotData.mean(), color='blue', alpha=0.3, linewidth=3)
#     ax.set_xlabel(plotVariable)
#     fig.show()

# basal:
# Final H in the whole population: 8.6312525335816
# Final E in the whole population: 1.276190476190476
# need:
# Final H in the whole population: 8.430695563232915
# Final E in the whole population: 1.3129251700680271
# risk :
# Final H in the whole population: 8.6312525335816
# Final E in the whole population: 1.276190476190476
# Patient centred:
#Final H in the whole population: 8.211404953576356
#Final E in the whole population: 1.353061224489796
# H_segmented:
# Final H in the whole population: 8.541262360847192
# Final E in the whole population: 1.264625850340136
