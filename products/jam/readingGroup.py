from fontTools.misc.textTools import caselessSort
from scipy.stats import alpha

from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.pathFinder import PathFinder

from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.client import Client
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

p = LinePlotter()

varsigma = 300
OBS_PERIOD = 3

def generateDistancesGroup(file: str, java_output: str, working_directory: str, collect=False,
                           OBS_PERIOD=100) -> TrajectoryDistances:
    if collect:
        collector = PathFinderCollector(java_output)
        collector.buildRuns_ecj(file)
    data = pd.read_csv(java_output + f'{file}_allRuns.csv')
    data.rename(columns={'Fitness': "H"}, inplace=True)
    # Pick the last 40 generations
    data = data.loc[data.shape[0] - 40:data.shape[0]]
    distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                    ENGINE_PATH=ENGINE_PATH,
                                    varsigma=varsigma, selectionName=file, minH=0, maxH=1000,
                                    OBS_PERIOD=OBS_PERIOD, oederByWindow=100,
                                    orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                    addParam={'Pi'})
    distances.set_grain(data.shape[0])
    return distances


def build_deciles_frame(data: pd.DataFrame) -> pd.DataFrame:
    # create the DataFrame
    output = pd.DataFrame(index=data.index)
    # Do for every time window
    for window, values in data.items():
        # Compute the deciles for this window
        deciles = np.quantile(values, np.linspace(start=0.1, stop=1, num=10))
        # construct a Serie with the decile values for each index
        order = []
        for value in values:
            for decile in range(0, len(deciles)):
                o = 1  # defaults to last decile
                if value <= deciles[decile]:
                    o = decile + 1
                    break
            order.append(o)  # assign decile
        s = pd.Series(data=order, index=values.index, name=int(window))
        # Construct a dataframe with the deciles series per window
        output = output.join(s)
    return output


def cummulativeStateVariable(data: dict) -> pd.DataFrame:
    """Compute the cummulative sum of a state variable per patient and save into the dictionary"""
    return data.cumsum(axis=1)


def diffStateVariable(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the difference of a state variable per patient and save into the dictionary"""
    return data.diff(axis=1)


def proportionC(data: pd.DataFrame, low_h_cut: float, high_h_cut: float) -> pd.Series:
    """Compute the proportion of patients who got appointment from the total asking for appointments"""
    C = data['SimpleC'].loc[(data['H']['100'] <= high_h_cut) & (data['H']['100'] >= low_h_cut)]
    B = data['SimpleB'].loc[(data['H']['100'] <= high_h_cut) & (data['H']['100'] >= low_h_cut)]
    return C.mask(cond=B != 1)


def waterMark(k: str) -> str:
    titles = {'H': 'Cum. Health', 'N': 'Cum. Need', 'T': 'Cum Treatment', 'E': 'Cum Expectation', 'C': 'Cum Contacts',
              'O': 'Cum Noise-exp', 'D': 'Cum Disease',
              'h': 'Inst. Health', 't': 'Inst. Treatment', 'b': 'Inst. % seeking', 'e': 'Inst. Expectation',
              'o': 'Inst. noise-exp', 'd': 'Inst. disease', 'c': 'Inst. % seeking got appointment'}
    return titles[k]


def onePlot(ax: mpl.axes.Axes, dd: pd.DataFrame, plotVariable: str, color: str, low_h_cut: float = None,
            high_h_cut: float = None,
            plotData: pd.DataFrame = None) -> mpl.axes.Axes:
    # ExtractPlotData
    if plotData is None:
        plotData = dd[plotVariable.strip('_diff').strip('_cum')].loc[
            (dd['H']['100'] <= high_h_cut) & (dd['H']['100'] >= low_h_cut)] \
            if plotVariable not in ['c'] else proportionC(data=dd, low_h_cut=low_h_cut, high_h_cut=high_h_cut)
    # Comput diff or cum if needed
    plotData = plotData.diff(axis=1) if '_diff' in plotVariable else plotData
    plotData = plotData.cumsum(axis=1) if '_cum' in plotVariable else plotData
    ax.plot([x for x in range(plotData.shape[1])], plotData.mean(), color=color, linewidth=3)
    ax.set_xlabel(plotVariable, fontsize=20)
    return ax


def populate_axe(ax_dict, low_h_cut: float = None, high_h_cut: float = None, sub_title: str = "",
                 filteredData: pd.DataFrame = None):
    """
    Helper to populate the graphs into axes and the arrows.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    cmap = mpl.colormaps['plasma']
    arrow = cmap(0.8)
    # kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    scale = 25
    lineScale = scale / 4.5
    t = 40
    waterSize = 25
    subtitleSize = 25
    for k, ax in ax_dict.items():
        if k in ['1']:
            AA = mpatches.FancyArrowPatch(posA=(0.5, 1), posB=(0.5, 0), mutation_scale=scale, color=arrow)
            ax.add_patch(AA)
            ax.set_axis_off()
        if k in ['3']:
            AA = ConnectionPatch(xyA=(0, 0.6), xyB=(0.0, 0.3), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.00, 0.3), xyB=(0.6, 0.3), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.85, 0), xyB=(0.85, 0.3), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.85, 0.3), xyB=(0.6, 0.3), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = mpatches.FancyArrowPatch(posA=(0.6, 0.3), posB=(0.6, 1), mutation_scale=scale, color=arrow)
            ax.add_patch(AA)
            ax.text(x=0.3, y=0.6, s="Prescription", size=t, horizontalalignment='center')
            ax.set_axis_off()
        if k == '7':
            AA = ConnectionPatch(xyA=(0.2, 1), xyB=(0.2, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.2, 0), xyB=(0.2, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = mpatches.FancyArrowPatch(posA=(0.2, 0.5), posB=(1, 0.5), mutation_scale=scale, color=arrow)
            ax.add_patch(AA)
            ax.text(x=0.6, y=0.6, s="Behaviour", size=t, horizontalalignment='center')
            ax.set_axis_off()
        if k == 'p':
            ax.text(x=0.5, y=0.3, s="Progression", size=t, horizontalalignment='center')
            AA = ConnectionPatch(xyA=(1, 0.5), xyB=(0.25, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.85, 0.9), xyB=(0.85, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = mpatches.FancyArrowPatch(posA=(0.25, 0.5), posB=(0.25, 1), hatch='o', mutation_scale=scale,
                                          color=arrow)
            ax.add_patch(AA)
            ax.set_axis_off()
        if k == 'l':
            ax.text(x=0.4, y=0.8, s="Allocation", size=t, horizontalalignment='center')
            AA = mpatches.FancyArrowPatch(posA=(0, 0.5), posB=(1, 0.5), hatch='o', mutation_scale=scale, color=arrow)
            ax.add_patch(AA)
            ax.set_axis_off()
        if k == '9':
            AA = ConnectionPatch(xyA=(0.86, 0.9), xyB=(0.86, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.86, 0.5), xyB=(0.2, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.76, 0.1), xyB=(0.76, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = ConnectionPatch(xyA=(0.3, 0.9), xyB=(0.3, 0.5), coordsA=ax.transData, lw=lineScale, color=arrow)
            ax.add_patch(AA)
            AA = mpatches.FancyArrowPatch(posA=(0.2, 0.5), posB=(0.2, 0), hatch='o', mutation_scale=scale, color=arrow)
            ax.add_patch(AA)
            ax.set_axis_off()
        if k == 'x':
            ax.text(x=0.5, y=1, s="E. fromation", size=t, horizontalalignment='center')
            ax.set_axis_off()

        # stateVariables = ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC', 'T', 'Disease', 'ExpNoise', 'InstExp']
        # Cummulative vars
        if k in ['H', 'N', 'T', 'E', 'C', 'O', 'D']:
            color = cmap(0.1)
            ax.text(0, 1, waterMark(k), transform=ax.transAxes, size=waterSize)
            k = 'T_cum' if k == 'T' else k
            # k = 'SimpleE' if k =='E' else k
            # k = 'obsE' if k =='E' else k
            k = 'SimpleC_cum' if k == 'C' else k
            k = 'ExpNoise_cum' if k == 'O' else k
            k = 'Disease_cum' if k == 'D' else k
            # plotVariables = ['H', 'N', 'N_diff', 'SimpleE', 'SimpleE_diff', 'SimpleB', 'SimpleB_cum', 'SimpleC',
            #                  'SimpleC_cum', 'T', 'T_cum', 'Disease'
            #     , 'Disease_cum', 'ExpNoise', 'ExpNoise_cum', 'InstExp']
            if k == "obsE":
                print(8)
            onePlot(ax, dd=dd, plotVariable=k, color=color, low_h_cut=low_h_cut, high_h_cut=high_h_cut,
                    plotData=filteredData)

            #     tik = ax.get_yticks() if 0 in ax.get_yticks() else np.append(0, ax.get_yticks())
            #     ax.set_yticks(ticks = tik, labels = tik)
            # ax.set_frame_on(False)

        # Dynamic vars
        if k in ['h', 't', 'b', 'e', 'o', 'd', 'c']:
            color = cmap(0.6)
            ax.text(0, 1, waterMark(k), transform=ax.transAxes, size=waterSize)
            k = 'T' if k == 't' else k
            k = 'SimpleB' if k == 'b' else k
            # k = 'SimpleE_diff' if k =='e' else k
            k = 'E_diff' if k == 'e' else k
            k = 'ExpNoise' if k == 'o' else k
            k = 'H_diff' if k == 'h' else k
            k = 'Disease' if k == 'd' else k
            onePlot(ax, dd=dd, plotVariable=k, color=color, low_h_cut=low_h_cut, high_h_cut=high_h_cut,
                    plotData=filteredData)
            # ax.set_frame_on(False)
        if k == 's':  # just for subtitle
            decileSize = dd['H'].loc[(dd['H']['100'] <= high_h_cut) & (dd['H']['100'] >= low_h_cut)].shape[0]
            ax.set_title(f'Decile size = {decileSize} {sub_title}', fontsize=subtitleSize)

            ax.set_axis_off()
        # MAKE SURE THESE GRAPHS START AT Y = 0
        if k in ['H'] and 0 not in ax.get_yticks():
            ax.set_yticks(np.linspace(start=0, stop=ax.get_yticks().max(), num=8).round(1))


def complexAxeDict():
    axd = plt.figure(layout="constrained", figsize=(30, 30)).subplot_mosaic(
        """
        sssss
        HhddD
        1pp..
        NpptT
        73333
        7blcC
        7999.
        EexoO
        """,
        gridspec_kw=dict(width_ratios=[1, 1, 0.5, 1, 1], height_ratios=[0.1, 1, 0.2, 1, 0.2, 1, 0.2, 1])
    )
    return axd


def jaccardCoeficient(data: dict, stateVar):
    """Compute a jaccard matrix of patients at each timestep.
    Parameters: data is a dic with seeds, each seeed a simulation"""
    counter = 0
    jaccards = np.empty((0, data[next(iter(data))][stateVar].shape[1], data[next(iter(data))][stateVar].shape[1]))
    for seed, simulation in data.items():
        jaccardMatrix = np.empty((0, simulation[stateVar].shape[1]))
        for windowA in simulation[stateVar]:
            jaccard = []
            for windowB in simulation[stateVar]:
                # For SimpleB
                JaccardSum = simulation[stateVar][windowA] + simulation[stateVar][windowB]
                if sum(JaccardSum > 0) == 0:  # this means there are no IDs, so no union, its an NA
                    jaccard.append(np.nan)
                else:
                    jaccard.append(sum((simulation[stateVar][windowB] > 0) & (simulation[stateVar][windowA] > 0)) / sum(
                        (simulation[stateVar][windowB] > 0) + (simulation[stateVar][windowA] > 0)))
            jaccardMatrix = np.concatenate((jaccardMatrix, np.array([jaccard])), axis=0)
        jaccards = np.concatenate((jaccards, np.array([jaccardMatrix])), axis=0)
        if counter == 0:
            break
        counter += 1
    return jaccards


def seekersExplorer(data: dict, stateVar, behaviour):
    """Measure stateVar in the temporal vecinity of patients that were seeking care.
    Parameters: data is a dic with seeds, each seeed a simulation"""
    counter = 0
    size = data[next(iter(data))][stateVar].shape[1]
    vecinities = np.empty((0, size, size))
    for seed, simulation in data.items():
        vecinityMatrix = np.empty((0, size))
        # for windowA in simulation[stateVar]:
        for windowA in range(size):
            match behaviour:
                case "seekers":
                    mask = simulation['SimpleB'].iloc[:, windowA] != 1  # use this mask to see continuity
                case "treated":
                    mask = simulation['T'].iloc[:, windowA] == 0
            vecinity = simulation[stateVar].mask(mask, other=np.nan).mean(axis=0)
            vecinityMatrix = np.concatenate((vecinityMatrix, np.array([vecinity])), axis=0)

        vecinities = np.concatenate((vecinities, np.array([vecinityMatrix])), axis=0)
        if counter == 0:
            break
        counter += 1
    return vecinities


def needsTreatmentHeat(data: dict, treatment: str):
    fig, heatAx = plt.subplots(figsize=(20, 10), constrained_layout=True)
    for index, simulation in data.items():
        finalNeed = simulation['N'].iloc[:, -1].round(1)
        orderedData = pd.DataFrame(data=simulation['T'].to_numpy(), index=finalNeed).sort_index(axis=0, ascending=False)
        heatAx = sns.heatmap(orderedData, cbar=True, cbar_kws={'pad': 0.1})
        heatAx.collections[0].colorbar.set_label("Intensity of treatment", labelpad=15, fontsize=20)

        heatAx.set_title(treatment)
        right = heatAx.twinx()
        right.yaxis.set_ticks_position('right')
        right.set_ylabel("Needs at the end of the simulation", rotation=90, labelpad=10, fontsize=20)
        # Copy tick positions and labels
        tick_positions = heatAx.get_yticks()  # Get tick positions from the left y-axis
        tick_labels = heatAx.get_yticklabels()  # Get tick labels from the left y-axis
        # Apply the same tick positions and labels to the right side
        right.set_yticks(tick_positions)  # Set positions for both sides
        right.set_yticklabels([label.get_text() for label in reversed(tick_labels)])  # Copy labels

        # put 0 in the left:
        newLab = [0] * len(tick_labels)
        heatAx.set_yticklabels(newLab)
        # heatAx.set_ylabel("Needs by the end of simulation", rotation=90, labelpad=10, fontsize=15)
        # heatAx.yaxis.tick_right()
        # heatAx.yaxis.set_label_position("right")
        # heatAx.tick_params(axis='y', labelrotation=0)
        heatAx.set_ylabel("Needs at the begining of the simulation", rotation=90, labelpad=10, fontsize=20)

        heatAx.set_xlabel("Time (cycles)", fontsize=15)

        fig.suptitle(
            f"Treeatment trajectories for all patients. Allocation: {tit_treatmentType(treatment)}. ({tit_systemType(treatment)})",
            size=25)
        fig.show()
        return


def generateErrorData(params, reps, stateVariables, initialSeed):
    indexOfSimulations = pd.DataFrame(columns=params.index.to_list().append("H"))
    ss = initialSeed
    for i in range(reps):
        newParams = {}
        for index, value in params.items():
            if value:  # not include false values
                newParams[index] = [value]
            if value == "True":
                newParams[index] = ["true"]
        newParams["obsH"] = True
        newParams["seeds"] = ss
        newParams['H'] = 8  ##Just to create a PathFinder
        indexOfSimulations = pd.concat([indexOfSimulations, pd.DataFrame(newParams)])
        ss = ss + 1485
    producer = PathFinder(working_directory=f'{working_directory}/errorBars/', ENGINE_PATH=ENGINE_PATH,
                          varsigma=params['varsigma'],
                          OBS_PERIOD=params['OBS_PERIOD'], allRuns_csv=indexOfSimulations)
    producer.createSelection(name=treatment, minH=0, maxH=1200)
    return producer.produce(selectionName=treatment, stateVariables=stateVariables)


def tit_systemType(treatment: str) -> str:
    return 'No-expectation system' if 'low' in treatment else 'system with expectations'


def tit_treatmentType(treatment: str) -> str:
    if 'basal' in treatment:
        return 'First-come, first-served'
    if 'risk' in treatment:
        return 'Risk-stratified'
    if 'need' in treatment:
        return 'Patient-reported needs'


def makeHist(errorData, jamHistAx, cmap=mpl.colormaps['plasma']):
    """Make the 3 H histograms with error bars"""

    histCounts = np.empty((0, 19))
    tailDescription = np.empty((0, 2))
    for key, value in errorData.items():
        a = np.array(errorData[key]['H'])[:, 100]  # last window, all patients
        counts, bins = np.histogram(a, bins=range(20))
        histCounts = np.concatenate((histCounts, np.array([counts])), axis=0)
        # compute some quantiles for the paper
        q = np.quantile(a, [0.25, 0.75])
        np.quantile(a, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        tailDescription = np.concatenate((tailDescription, np.array([q])), axis=0)
    print(f'TAIL DESCRIPTION for {treatment}')
    print(f'quentiles 0.25 and 0.75 means {tailDescription.mean(0)}')
    print(f'lower quartile 0.05 and 0.95: {np.quantile(tailDescription[:, 0], q=[0.05, 0.95])}')
    print(f'upper quartile 0.05 and 0.95: {np.quantile(tailDescription[:, 1], q=[0.05, 0.95])}')
    if treatment == 'hiPsi/basal' or treatment == 'lowPsi/basal' or treatment == 'fig1/basal':
        axNum, tit = 0, "First-Come, First-Served Scheduling"
    elif treatment == 'hiPsi/risk' or treatment == 'lowPsi/risk' or treatment == 'fig1/risk':
        axNum, tit = 1, "Risk-Scheduling"
    elif treatment == 'hiPsi/need' or treatment == 'lowPsi/need' or treatment == 'fig1/need':
        axNum, tit = 2, "Patient-Reported Outcomes Scheduling"
    lowerErrors = np.quantile(histCounts, q=0.05, axis=0)
    upperErrors = np.quantile(histCounts, q=0.95, axis=0)
    x_error_pos = np.arange(start=0.5, stop=19, step=1)
    jamHistAx[axNum].stairs(histCounts.mean(0), bins, fill=True, color=cmap(0.6 - (axNum * 0.3)))
    jamHistAx[axNum].errorbar(x=x_error_pos, y=histCounts.mean(0),
                              yerr=np.absolute(np.quantile(histCounts, q=[0.05, 0.95], axis=0) - histCounts.mean(0)),
                              fmt='.')
    jamHistAx[axNum].set_ylim(top=1200)
    jamHistAx[axNum].set_title(tit, fontsize=15)
    jamHistAx[axNum].set_xlabel("Number of Unserved Needs per Patient", fontsize=15)
    jamHistAx[axNum].set_ylabel("Number of Patients", fontsize=15)
    # jamHist.suptitle(f"Distribution of unserved needs by the end of the simulations. {tit_systemType(treatment)}. {'lambda = ' + str(tweakedLambda) if tweakedLambda is not None else '' }", size = 15)


# jamFigure, jamAxes = plt.subplots(nrows=3, ncols= 3, figsize=(40, 30), constrained_layout=True, facecolor='white',)
jamHist, jamHistAx = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), constrained_layout=False, facecolor='white', )
# Participation and treatments grid
jamTime, jamTimeAx = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True, facecolor='white', )
# Participation and treatments in one figure
jamTime2, jamTimeAx2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), constrained_layout=True, facecolor='white', )
jamTimeAx2twinx = jamTimeAx2.twinx()
# Final JAM Figure
finalJamFigure, finalJamAxes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True,
                                            facecolor='white', )
finalJamAxesTwinx = finalJamAxes[1].twinx()

# treatments = ['hiPsi/basal','hiPsi/risk','hiPsi/need','lowPsi/basal','lowPsi/risk','lowPsi/need']
treatments = ['fig1/basal','fig1/risk','fig1/need']
# treatments = ['lowPsi/basal','lowPsi/risk','lowPsi/need']
SimpleBJaccard, TJaccard, sN, tN = {}, {}, {}, {}
cmap = mpl.colormaps['plasma']
for treatment in treatments:
    ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
    java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
    # working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'
    working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/'
    varsigma = varsigma
    # treatment = 'lowPsi/need'
    data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/{treatment}.csv')
    selection = treatment
    distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                    ENGINE_PATH=ENGINE_PATH,
                                    varsigma=varsigma, selectionName=selection, minH=0, maxH=1000,
                                    OBS_PERIOD=OBS_PERIOD, oederByWindow=OBS_PERIOD*2,
                                    orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                    addParam={'Pi'})
    # stateVariables= ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise', 'InstExp']
    stateVariables = ['H', 'N', 'E', 'SimpleB', 'SimpleC', 'T', 'Disease', 'ExpNoise', 'InstExp']

    # seed = 159753458941
    # seed = 136262753023 # no noise seed
    seed = 62920828945

    # dd = distances.produce(stateVariables=stateVariables, selectionName='dynamicLine', seeds=[62920828944])[62920828944]
    # tweak={'W':1})[int(seedWithMaxFitnes.iloc[0])]
    dd = distances.produce(stateVariables=stateVariables, selectionName=selection, seeds=[seed])[seed]
    tweakedLambda = None
    # dd=distances.produce(stateVariables=stateVariables, selectionName=selection, seeds=[seed], tweak={'fixed_lambda': tweakedLambda})[seed]
    params = p.printableParams(distances.get_seedParams(seed=seed, selectionName=selection, filterParams=False),
                               paramsPerLine=8, policy=True)

    ############################
    # Getting error bars

    ##Producing the lines rep times with different seeds:
    parameters_recover = distances.get_seedParams(seed=seed, selectionName=selection, filterParams=False)
    parameters_recover['seeds'] = parameters_recover['seed']
    # parameters_recover['varsigma'] = 300

    # producer.makedir(selectionName=treatment)

    # errorData = {0:dd}
    errorData = generateErrorData(params=parameters_recover, reps=100, stateVariables=stateVariables,
                                  initialSeed=52920828945)

    makeHist(errorData=errorData, jamHistAx=jamHistAx)

    # needsTreatmentHeat(data = errorData, treatment = treatment)

    if treatment == 'hiPsi/basal' or treatment == 'lowPsi/basal' or treatment == 'fig1/basal':
        axNum, tit, l, ll = 0, "First-Come, First-Served Scheduling", ["dotted", "dotted", "dotted", "dotted"], [
            "solid", "dotted", "solid", "dotted"]
    elif treatment == 'hiPsi/risk' or treatment == 'lowPsi/risk' or treatment == 'fig1/risk':
        axNum, tit, l, ll = 1, "Risk-Scheduling", ["dashed", "dashed", "dashed", "dashed"], ["dashdot", "dashed",
                                                                                             "dashdot", "dashed"]
    elif treatment == 'hiPsi/need' or treatment == 'lowPsi/need' or treatment == 'fig1/need':
        axNum, tit, l, ll = 2, "Patient-Reported Outcomes Scheduling", ["solid", "solid", "solid", "solid"], [
            (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))]

    ### Compara Treatments and participation
    # participation

    x_error_pos = np.arange(start=0, stop=101, step=1)

    # compute the jaccards:
    SimpleBJaccard[treatment] = jaccardCoeficient(data=errorData, stateVar='SimpleB')
    print(f'Simple B jaccard {treatment} = {np.nanmean(SimpleBJaccard[treatment])}')
    TJaccard[treatment] = jaccardCoeficient(data=errorData, stateVar='T')
    print(f'TJaccard jaccard {treatment} = {np.nanmean(TJaccard[treatment])}')
    # compute the seeker explorers:
    sN[treatment] = seekersExplorer(data=errorData, stateVar='N', behaviour="seekers")
    tN[treatment] = seekersExplorer(data=errorData, stateVar='N', behaviour="treated")

    seekMean = np.empty((0, 101))
    treatMean = np.empty((0, 101))
    needMean = np.empty((0, 101))

    ## DO only one line directly from dd. Useful for seeing the effect of tweaking when no errors repetitions have been recomputed
    # errorData = {}
    # errorData[0] = dd

    for key, value in errorData.items():
        # seek
        a = np.array(errorData[key]['SimpleB']).mean(0)  # last window, all patients
        seekMean = np.concatenate((seekMean, np.array([a])), axis=0)
        # treat
        b = np.array(errorData[key]['T'])
        np.putmask(b, b == 0, np.nan)  # not consider 0 treatments
        bb = np.nanmean(b, axis=0)
        treatMean = np.concatenate((treatMean, np.array([bb])), axis=0)
        # need
        c = np.array(errorData[key]['N']).mean(0)
        needMean = np.concatenate((needMean, np.array([c])), axis=0)
    # for key, value in errorData.items():
    #     a = np.array(errorData[key]['SimpleB']).mean(0) # last window, all patients
    #     seekMean = np.concatenate((seekMean, np.array([a])), axis=0)
    # seekAxes = [jamTimeAx[0], jamTimeAx2]

    ### Produce graph directly from dd withou errors
    # seekMean = dd['SimpleB']
    # treatMean = dd['T']
    # needMean = dd['N']

    ##Confiure graphs
    # Grid means a grd of two figures, double is needs+treatments in one axe
    seekAxes = {'grid': [jamTimeAx[1], ',', True, tit, l.copy()],
                'double': [jamTimeAx2, ',', False, f'Care-seeking, {tit}',
                           ll.copy()]}  # The list contains: 0 axe, 1 fmt, 2 wheter to call legend, 3 legend text, 4 linestyle
    treatAxes = {'grid': [jamTimeAx[0], ',', True, tit, l.copy()],
                 'double': [jamTimeAx2twinx, ',', False, f'Treatment, {tit}',
                            ll.copy()]}  # The list contains: 0 axe, 1 fmt, 2 wheter to call legend, 3 legent text, 4 linestyle
    # treatAxes = None #The list contains: 0 axe, 1 fmt, 2 wheter to call legend, 3 legent text, 4 linestyle
    # needsAxes = {'grid': [jamTimeAx[1], ',', True, tit, l], 'double': [jamTimeAx2twinx, ',', False, f'Need, {tit}', ll]}
    needsAxes = None

    for key, seekAxis in seekAxes.items():
        seekAxis[0].plot(range(101), seekMean.mean(0), label=seekAxis[3], linestyle=seekAxis[4].pop(),
                         color=cmap(0.6 - (axNum * 0.3)))
        seekAxis[0].errorbar(x=x_error_pos, y=seekMean.mean(0),
                             yerr=np.absolute(np.quantile(seekMean, q=[0.05, 0.95], axis=0) - seekMean.mean(0)),
                             fmt=seekAxis[1],
                             ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))

        seekAxis[0].legend() if seekAxis[2] else None
        seekAxis[0].set_xlabel("Simulation time (cycles)")
        seekAxis[0].set_ylabel("Proportion of patients asking for appointments")
        seekAxis[0].set_title("Population care-seeking behaviour") if seekAxis[2] else None

    if treatAxes is not None:
        for key, treatAxe in treatAxes.items():
            treatAxe[0].plot(range(101), treatMean.mean(0), label=treatAxe[3], linestyle=treatAxe[4].pop(),
                             color=cmap(0.6 - (axNum * 0.3)))
            treatAxe[0].errorbar(x=x_error_pos, y=treatMean.mean(0),
                                 yerr=np.absolute(np.quantile(treatMean, q=[0.05, 0.95], axis=0) - treatMean.mean(0)),
                                 fmt=treatAxe[1],
                                 ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))
            treatAxe[0].legend() if treatAxe[2] else None
            treatAxe[0].set_xlabel("Simulation time (cycles)")
            treatAxe[0].set_ylabel("Average needs solved per allocated patient")
            treatAxe[0].set_title("Treatment") if treatAxe[2] else None

    if needsAxes is not None:
        for key, needsAxe in needsAxes.items():
            needsAxe[0].plot(range(101), needMean.mean(0), label=needsAxe[3], linestyle=needsAxe[4].pop(),
                             color=cmap(0.6 - (axNum * 0.3)))
            needsAxe[0].errorbar(x=x_error_pos, y=needMean.mean(0),
                                 yerr=np.absolute(np.quantile(needMean, q=[0.05, 0.95], axis=0) - needMean.mean(0)),
                                 fmt=needsAxe[1],
                                 ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))
            needsAxe[0].legend() if needsAxe[2] else None
            needsAxe[0].set_xlabel("Simulation time (cycles)")
            needsAxe[0].set_ylabel("Average needs per patient (treatment)")
            needsAxe[0].set_title("Needs") if needsAxe[2] else None
    suptitle = (f"Temporal trajectories of participation and {'treatment' if treatAxes is not None else 'needs'} "
                f"{tit_systemType(treatment)} {'lambda = ' + str(tweakedLambda) if tweakedLambda is not None else ''}")
    jamTime.suptitle(suptitle, size=15)
    jamTime2.suptitle(suptitle, size=15)
    # Put legend for double axe
    handlesS, labelsS = jamTimeAx2.get_legend_handles_labels()
    handlesT, labelsT = jamTimeAx2twinx.get_legend_handles_labels()
    print(8)
    jamTimeAx2.legend(handles=handlesS + handlesT, labels=labelsS + labelsT, loc='lower right', frameon=False)

    ########
    ### For he final JAM figure
    # Perfornamce alone on the left
    finalAxConfigLEFT = ['', ',', False, f'{tit}', l.copy()]
    finalJamAxes[0].plot(range(101), treatMean.mean(0), label=finalAxConfigLEFT[3],
                         linestyle=finalAxConfigLEFT[4].pop(), color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxes[0].errorbar(x=x_error_pos, y=treatMean.mean(0),
                             yerr=np.absolute(np.quantile(treatMean, q=[0.05, 0.95], axis=0) - treatMean.mean(0)),
                             fmt=finalAxConfigLEFT[1],
                             ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxes[0].legend()
    finalJamAxes[0].set_xlabel("Simulation Time (Cycles)", fontsize=12)
    finalJamAxes[0].set_ylabel("Average Needs Solved per Patient During a Visit", fontsize=12)
    finalJamAxes[0].set_title("Performance", fontsize=15)

    # Needs and seeking on the right
    # Needs
    finalAxConfigRIGHT = ['', ',', False, f'Needs, {tit.replace("Scheduling", "")}', ll.copy()]
    finalJamAxes[1].plot(range(101), needMean.mean(0), label=finalAxConfigRIGHT[3], linestyle="solid",
                         color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxes[1].errorbar(x=x_error_pos, y=needMean.mean(0),
                             yerr=np.absolute(np.quantile(needMean, q=[0.05, 0.95], axis=0) - needMean.mean(0)),
                             fmt=finalAxConfigRIGHT[1],
                             ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxes[1].set_xlabel("Simulation Time (Cycles)", fontsize=12)
    finalJamAxes[1].set_ylabel("Average Unsolved Needs per Patient", fontsize=12)
    finalJamAxes[1].set_title("Population Needs and Seeking Behaviour", fontsize=15)
    # Participation
    finalJamAxesTwinx.plot(range(101), seekMean.mean(0), label=f'Seeking {tit.replace("Scheduling", "")}',
                           linestyle="dotted",
                           color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxesTwinx.errorbar(x=x_error_pos, y=seekMean.mean(0),
                               yerr=np.absolute(np.quantile(seekMean, q=[0.05, 0.95], axis=0) - seekMean.mean(0)),
                               fmt=finalAxConfigRIGHT[1],
                               ecolor=cmap(0.6 - (axNum * 0.3)), elinewidth=0.2, color=cmap(0.6 - (axNum * 0.3)))
    finalJamAxesTwinx.set_ylabel("Proportion of Patients Asking for Appointments", fontsize=12)
    finalJamAxes[1].set_ylim(bottom=0, top=7)
    handlesNedds, labelsNeeds = finalJamAxes[1].get_legend_handles_labels()
    handlesPart, labelsPart = finalJamAxesTwinx.get_legend_handles_labels()
    finalJamAxes[1].legend(handles=handlesNedds + handlesPart, labels=labelsNeeds + labelsPart, loc='lower right',
                           frameon=True)

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
    # Print separated graphs for paper figure
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
    # allData = np.quantile(np.array(dd['H'][f'{varsigma}']), q=[1])[0]
    # populate_axe(complexAxeDict(),low_h_cut = 0, high_h_cut = allData,sub_title = params)
    # print(f'Final H in the whole population: {dd["H"][f'{varsigma}'].mean()}')
    # print(f'Final E in the whole population: {dd["E"][f'{varsigma}'].mean()}')
    # plt.suptitle(f'All population seed: {seed} {treatment}', fontsize=50)
    # plt.show()
    #    print(f'Final E in the whole population: {dd["SimpleE"][f'{varsigma}'].mean()}')
    # fig, ax = plt.subplots(1,1, figsize=(5, 5), constrained_layout=True, facecolor='ghostwhite',)
    # ax.hist(dd['H']['100'], bins=50)
    # fig.suptitle(f'Health status at the end of the simulation {treatment}')
    # fig.show()

    # Coupling of needs and exp
    import seaborn as sns

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True, facecolor='ghostwhite', )
    # ax.plot(dd['N'].to_numpy().flatten(), dd['E'].to_numpy().flatten(), alpha = 0.3, color = 'blue')
    # ax.set_xlabel('N')
    # ax.set_ylabel('E')
#    print(f'correlation E and N in {treatment}: {np.corrcoef(dd['N'].to_numpy().flatten(), dd['E'].to_numpy().flatten())}')
############################
# Dynamics per decile
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

# jamAxes[0,0].set_yticks([590], labels=["first-served"], fontsize = 50)
# make_axes_area_auto_adjustable(jamAxes)
# jamFigure.show()
jamHist.show()
jamTime.show()
jamTime2.show()
finalJamFigure.show()
# stateVariables= ['InstExp']
# dd = dist_inequal.produce(stateVariables=stateVariables, selectionName='inequal', seeds=seedWithMaxFitnes,
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
#     plotData = plotData.cumsum(axis=1) if '_cum' in plotVariable else plotData
#     #for index, row in plotData.iterrows():
#     #    ax.plot([x for x in range(len(row))], row, color='blue', alpha=0.02)
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
# Final H in the whole population: 8.211404953576356
# Final E in the whole population: 1.353061224489796
# H_segmented:
# Final H in the whole population: 8.541262360847192
# Final E in the whole population: 1.264625850340136

# fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize = (15,5))
# axNum= 0


##### The Jaccard Matrices:
for whichMatrix in ["B", "T", "sN", "tN"]:
    # whichMatrix = "B" # B for seeking or any string for treatment
    for treatment in treatments:
        match whichMatrix:
            case "B":
                arr, title, subtitle = (SimpleBJaccard[treatment][0], "Jaccard seek-care index among cycles",
                                        "Computes the similary in seeking behaviour as \n the proportion of identical patient who sought care between cycles.")
            case "T":
                arr, title, subtitle = (TJaccard[treatment][0], "Jaccard treatment index among cycles",
                                        "Computes the similary in receiving treatment as \n the proportion of identical patient who received treatment between cycles.")
            case "sN":
                arr, title, subtitle = (sN[treatment][0], "Needs tracker for patients seeking care",
                                        "Computes the average needs for patients seeking care in the current cycle (diagonal) \n and their average needs for each simulation cycle.")
            case "tN":
                arr, title, subtitle = (tN[treatment][0], "Needs tracker for patients who got treatment",
                                        "Computes the average needs for patients receiving treatment in the current cycle (diagonal) \n and their average needs for each simulation cycle.")

        heatAx = sns.heatmap(arr)
        f, a = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), constrained_layout=True, facecolor='white', )
        a.plot(range(0, 101), arr.diagonal(), color='black')
        a.plot(range(1, 101), arr.diagonal(offset=1), color='blue')
        a.plot(range(2, 101), arr.diagonal(offset=2), color='red')
        a.set_title("Black diagonal, Blue +1, Red +2 cycles")
        f.suptitle(f"{whichMatrix} {tit_treatmentType(treatment)} {tit_treatmentType(treatment)}")
        f.show()
        heatAx.set_title(subtitle, size=7)
        plt.suptitle(f'{title} in {treatment.split('/')[1]}.'
                     f'{tit_systemType(treatment)}')
        # plt.suptitle(f'{treatment}')
        plt.xlabel('Simulation cycle (time)')
        plt.ylabel('Simulation cycle (time)')
        axNum += 1
        plt.show()
        arr.diagonal(offset=-3)
        print(f'Jaccard_{whichMatrix} index between current and previous (-) and future (+) cycles. \n'
              f'{treatment}\n'
              f'-4 steps: {arr.diagonal(offset=-4).mean().round(2)} \n'
              f'-3 steps: {arr.diagonal(offset=-3).mean().round(2)} \n'
              f'-2 steps: {arr.diagonal(offset=-2).mean().round(2)} \n'
              f'-1 steps: {arr.diagonal(offset=-1).mean().round(2)} \n'
              f'-----\n'
              f'+1 steps: {arr.diagonal(offset=1).mean().round(2)} \n'
              f'+2 steps: {arr.diagonal(offset=2).mean().round(2)} \n'
              f'+3 steps: {arr.diagonal(offset=3).mean().round(2)} \n'
              f'+4 steps: {arr.diagonal(offset=4).mean().round(2)} \n'
              )

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

# jamAxes[0, 0].set_yticks([550, 650], labels=["first-served ", "First-come "], fontsize=50)
# jamAxes[1, 0].set_yticks([600], labels=["Risk-stratified "], fontsize=50)
# jamAxes[2, 0].set_yticks([650, 550], labels=["Patient-reported ", "needs "], fontsize=50)
