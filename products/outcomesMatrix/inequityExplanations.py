#from zipimport import alt_path_sep

from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.trajectoryDistances import TrajectoryDistances
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'

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

def cummulativeStateVariable(data: dict, stateVar: str) -> pd.DataFrame:
    """Compute the cummulative sum of a state variable per patient and save into the dictionary"""

    data[f'cum_{stateVar}'] = data[stateVar].cumsum(axis=1)
    return data

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

def paint(decile: int, cmap = mpl.colormaps['plasma']) -> list:
    return cmap(0.7) if decile > 1 else cmap(0), 0 if decile > 1 else 0.3

def purity(serie):
    return serie == serie[100]

dist_inequal = generateDistancesGroup(file="inequal", java_output=java_output, working_directory=working_directory,
                                          OBS_PERIOD=1)
seedWithMaxFitnes = dist_inequal.selections['inequal'].loc[
        dist_inequal.selections['inequal']['H'] == dist_inequal.selections['inequal']['H'].max()]['seeds']
dd_w1 = dist_inequal.produce(stateVariables='H', selectionName='inequal', seeds=seedWithMaxFitnes)[
        int(seedWithMaxFitnes.iloc[0])]
decilesFrame = build_deciles_frame(dd_w1['H'])
purityDeciles = decilesFrame.apply(purity, axis = 1)

##Try observe disease and expNoise:

Disease = dist_inequal.produce(stateVariables=['Disease'], selectionName='inequal', seeds=seedWithMaxFitnes)[
        int(seedWithMaxFitnes.iloc[0])]
ExpNoise = dist_inequal.produce(stateVariables=['ExpNoise'], selectionName='inequal', seeds=seedWithMaxFitnes)[
        int(seedWithMaxFitnes.iloc[0])]


##Instantaneous T:
# LowT, UpT = dd['T'].loc[dd['H']['100'] <= lowerDecile], dd['T'].loc[dd['H']['100'] >= upperDecile]
# LowC, UpC = dd['SimpleC'].loc[dd['H']['100'] <= lowerDecile], dd['SimpleC'].loc[dd['H']['100'] >= upperDecile]
# LowCondition = LowC == 0
# for index, row in LowT.mask(LowCondition).iterrows():
#     ax[0][1].plot([x for x in range(101)], row, color='blue', alpha=0.1)
# ax[0][1].plot([x for x in range(101)], LowT.mask(LowCondition).mean(0), color='blue', alpha=0.3, linewidth = 3)
##Instantaneous Disease:
#ax[2][2].plot([x for x in range(100)], LowDisease.iloc[:,0:100].mean(0), color='purple', alpha=0.3, linewidth = 3)

#Comparison of trajectories for extreme deciles for equity and inequity
##TODO: if this code result useul, it would be better to integrate with above
p = LinePlotter()
tweakVar = 'fixed_lambda, W'
# tweakValues = [20,20]
#tweakValues = [{'fixed_lambda': 20, 'W':1}, {'fixed_lambda': 20, 'W':20}]
tweakValues = [{ 'W':1}, { 'W':20}]

stateVariables=         ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T']
#stateVariables=         ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise']

plotStateVariables =    ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','cum_T']
fig, ax = plt.subplots(ncols=len(stateVariables), nrows=len(tweakValues)+2, figsize=(5*len(stateVariables), 5*(len(tweakValues)+2)), constrained_layout=True, facecolor='ghostwhite',)
for value in range(0,len(tweakValues)):
    dd = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes, tweak = tweakValues[value])[int(seedWithMaxFitnes.iloc[0])]
                              #tweak={f'{tweakVar}': tweakValues[value]})[int(seedWithMaxFitnes.iloc[0])]
    if value ==0:
        print(0)
    dd = cummulativeStateVariable(data=dd, stateVar='T')
    lowerDecile, upperDecile = np.quantile(np.array(dd['H']['100']), q=[0.1,0.9])
    for plotStateVariable in range(0,len(plotStateVariables)):
        if plotStateVariables ==3:
            print(0)
        ddLow, ddUp = dd[plotStateVariables[plotStateVariable]].loc[dd['H']['100']<= lowerDecile], dd[plotStateVariables[plotStateVariable]].loc[dd['H']['100']>= upperDecile]
        for index, row in ddLow.iterrows():
            ax[value*2][plotStateVariable].plot([x for x in range(101)], row, color = 'blue', alpha = 0.02)
            ax[value*2][plotStateVariable].set_xlabel(stateVariables[plotStateVariable])
        ax[value*2][plotStateVariable].plot([x for x in range(101)], ddLow.mean(), color = 'blue', alpha = 0.3,  linewidth = 3)
        for index, row in ddUp.iterrows():
            ax[value*2+1][plotStateVariable].plot([x for x in range(101)], row, color = 'blue', alpha = 0.02)
            ax[value*2+1][plotStateVariable].set_xlabel(plotStateVariables[plotStateVariable])
        ax[value*2+1][plotStateVariable].plot([x for x in range(101)], ddUp.mean(), color='blue', alpha=0.3, linewidth=3)
    #for i in range(0, len(plotStateVariables)):
    ax[0,2].set_title(f'Lower decile {tweakVar} = {tweakValues[0]} ', size=20)
    ax[1,2].set_title(f'Upper decile {tweakVar} = {tweakValues[0]} ', size=20)
    ax[2,2].set_title(f'Lower decile {tweakVar} = {tweakValues[1]}', size=20)
    ax[3,2].set_title(f'Upper decile {tweakVar} = {tweakValues[1]}', size=20)
fig.show()

##Evolution of distributions of C for best and wors H with W=1
# p = LinePlotter()
# W = [1]
# for w in range(0,len(W)):
#     time = [0,10,20,30,40,50,60,70,80,90,100]
#     #time = [0,20,40,60,80,100]
#     stateVariables=['SimpleC']
#     fig, ax = plt.subplots(ncols=len(time), nrows=len(stateVariables), figsize=(5*len(time),30), constrained_layout=True, facecolor='ghostwhite',)
#     dd = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
#                               tweak={'W': W[w]})[int(seedWithMaxFitnes.iloc[0])]
#     #TODO separate here upper and lower deciles
#     for t in range(0,len(time)):
#         for var in range(0,len(stateVariables)):
#              p.hist(data = dd, stateVar=stateVariables[var], window=time[t], axe = ax[var][t],
#                     x_texsize = 20)
#     for row in range(0,len(stateVariables)):
#         p.same_limits_y(ax[row])
#         p.same_limits_x(ax[row])
#     for t in range(0,len(time)):
#         ax[0][t].set_title(f"Timestep {time[t]}", size = 30)
#
#     fig.suptitle(f"W = {W[w]}", size=40)
#     fig.show()

#Decile-trajectories for the lower decile
fig, ax = plt.subplots()
for index, row in decilesFrame.iterrows():
    ax.plot(decilesFrame.columns, np.random.normal(loc = row, scale = 0.05), color = paint(row[100])[0], alpha = paint(row[100])[1], linewidth = 0.11)
#ax.set_yticks(ticks=np.arange(1, 11, 1), labels=np.quantile(a = dd_w1['H']['100'], q = np.linspace(start=0.1, stop=1, num=10)).round(1))
ax.set_ylabel('Deciles of the H distribution')
ax.set_xlabel('Simulation steps')
fig.suptitle(f'Decile-trajectory of patients that ended with H in the first decile (H < {np.quantile(a = dd_w1['H']['100'], q=0.1).round(1)})')
#ax.set_xticks(ticks=np.arange(start=0, stop = 101, step=1), labels=)
fig.show()
print('stop')
##Mark an histogram with deciles information

#Purity of the lower decile
fig1, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 7), constrained_layout=True, facecolor='ghostwhite', )
for decile in np.arange(start=1, stop=11, step = 1):
    axes[0].plot(purityDeciles.loc[decilesFrame[100] == decile].sum(0), label = decile)
    axes[1].plot(purityDeciles.loc[decilesFrame[100] == decile].sum(0)/purityDeciles.loc[decilesFrame[100] == decile].shape[0], label = decile)
axes[0].legend(title = 'Deciles')
axes[1].legend(title = 'Deciles')
axes[0].set_ylabel('Number of patients')
axes[1].set_ylabel('Proportion of patients')
axes[0].set_xlabel('Simulation steps')
axes[1].set_xlabel('Simulation steps')
fig1.suptitle("Purity of deciles during the simulation")
fig1.show()


## Show that learning is key for W1 / W20 differences
p = LinePlotter()
W = [1, 20]
for w in range(0, len(W)):
    time = [100]
    # time = [0,20,40,60,80,100]
    stateVariables = ['H']
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 11), constrained_layout=True, facecolor='ghostwhite', )
    dd_w1 = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes)[
        int(seedWithMaxFitnes.iloc[0])]
    dd_w20 = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
                                  tweak={'W': 20})[int(seedWithMaxFitnes.iloc[0])]
    dd_w20_lambda20 = \
    dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
                         tweak={'W': 20, 'fixed_lambda': 20})[int(seedWithMaxFitnes.iloc[0])]
    dd_w1_lambda20 = \
    dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
                         tweak={'W': 1, 'fixed_lambda': 20})[int(seedWithMaxFitnes.iloc[0])]
    p.hist(data=dd_w1, stateVar='H', window='100', axe=ax[0][0], x_texsize=10)
    p.hist(data=dd_w20, stateVar='H', window='100', axe=ax[0][1], x_texsize=10)
    p.hist(data=dd_w1_lambda20, stateVar='H', window='100', axe=ax[1][0], x_texsize=10)
    p.hist(data=dd_w20_lambda20, stateVar='H', window='100', axe=ax[1][1], x_texsize=10)

    ax[0][0].set_title('\n N° Providers = 1, slow learning', size = 20)
    ax[0][1].set_title('\n N° Providers =20, slow learning', size = 20)
    ax[1][0].set_title('\n N° Providers =1, fast learning', size = 20)
    ax[1][1].set_title('\n N° Providers =20, fast learning', size = 20)

    ax[0][0].set_xlabel('(good)      OUTCOMES       (bad)', size = 18)
    ax[0][1].set_xlabel('(good)      OUTCOMES       (bad)', size = 18)
    ax[1][0].set_xlabel('(good)      OUTCOMES       (bad)', size = 18)
    ax[1][1].set_xlabel('(good)      OUTCOMES       (bad)', size = 18)

    ax[0][0].set_ylabel('Number of patients', size=18)
    ax[0][1].set_ylabel('Number of patients', size=18)
    ax[1][0].set_ylabel('Number of patients', size=18)
    ax[1][1].set_ylabel('Number of patients', size=18)

    ax[0] = p.same_limits_y(ax[0])
    ax[0] = p.same_limits_x(ax[0])
    ax[1] = p.same_limits_y(ax[1])
    ax[1] = p.same_limits_x(ax[1])

    fig.suptitle(f"The number of providers affects outcomes when learning is involved", size=25)
    fig.show()

##Evolution of distributions for each W
p = LinePlotter()
W = [1,20]
for w in range(0,len(W)):
    time = [0,10,20,30,40,50,60,70,80,90,100]
    #time = [0,20,40,60,80,100]
    stateVariables=['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T']
    fig, ax = plt.subplots(ncols=len(time), nrows=len(stateVariables), figsize=(5*len(time),30), constrained_layout=True, facecolor='ghostwhite',)
    dd = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
                              tweak={'W': W[w], 'fixed_lambda': 20})[int(seedWithMaxFitnes.iloc[0])]
    # Comparison of trajectories for extreme deciles for equity and inequity
    ##TODO Maybe call the trjectories as an optional function from here
    for t in range(0,len(time)):
        for var in range(0,len(stateVariables)):
             p.hist(data = dd, stateVar=stateVariables[var], window=time[t], axe = ax[var][t],
                    x_texsize = 20)
    for row in range(0,len(stateVariables)):
        p.same_limits_y(ax[row])
        p.same_limits_x(ax[row])
    for t in range(0,len(time)):
        ax[0][t].set_title(f"Timestep {time[t]}", size = 30)

    fig.suptitle(f"W = {W[w]}", size=40)
    fig.show()



# ## Show that learning is key for W1 / W20 differences
# p = LinePlotter()
# W = [1, 20]
# for w in range(0, len(W)):
#     time = [100]
#     # time = [0,20,40,60,80,100]
#     stateVariables = ['H']
#     fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 11), constrained_layout=True, facecolor='ghostwhite', )
#     dd_w1 = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes)[
#         int(seedWithMaxFitnes.iloc[0])]
#     dd_w20 = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
#                                   tweak={'W': 20})[int(seedWithMaxFitnes.iloc[0])]
#     dd_w20_lambda20 = \
#     dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
#                          tweak={'W': 20, 'fixed_lambda': 20})[int(seedWithMaxFitnes.iloc[0])]
#     dd_w1_lambda20 = \
#     dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
#                          tweak={'W': 1, 'fixed_lambda': 20})[int(seedWithMaxFitnes.iloc[0])]
#     p.hist(data=dd_w1, stateVar='H', window='100', axe=ax[0][0], x_texsize=10)
#     p.hist(data=dd_w20, stateVar='H', window='100', axe=ax[0][1], x_texsize=10)
#     p.hist(data=dd_w1_lambda20, stateVar='H', window='100', axe=ax[1][0], x_texsize=10)
#     p.hist(data=dd_w20_lambda20, stateVar='H', window='100', axe=ax[1][1], x_texsize=10)
#
#     ax[0][0].set_title('W=1, learning=0.5', size = 20)
#     ax[0][1].set_title('W=20, learning=0.5', size = 20)
#     ax[1][0].set_title('W=1, learning=20', size = 20)
#     ax[1][1].set_title('W=20, learning=20', size = 20)
#
#     ax[0] = p.same_limits_y(ax[0])
#     ax[0] = p.same_limits_x(ax[0])
#     ax[1] = p.same_limits_y(ax[1])
#     ax[1] = p.same_limits_x(ax[1])
#
#     fig.suptitle(f"The effect of W depends on lambda (learning)", size=25)
#     fig.show()

def NplusE(data: pd.DataFrame, lowerDecile: float, upperDecile: float) -> tuple:
    return data['SimpleE'].loc[data['H']['100']<= lowerDecile], data['SimpleE'].loc[data['H']['100']>= upperDecile]


