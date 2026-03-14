import matplotlib.axes
from matplotlib import pyplot as plt

from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.linePlotter import LinePlotter
import pandas as pd
import numpy as np

p = LinePlotter()

def build_deciles_frame(data: pd.DataFrame, method = 'inverted_cdf') -> pd.DataFrame:
    #create the DataFrame
    output = pd.DataFrame(index = data.index)
    #Do for every time window
    for window, values in data.items():
        #Compute the deciles for this window
        deciles = np.quantile(a = values, q = np.linspace(start=0, stop=1, num = 10), method = method)
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

def compute_transition(data: pd.DataFrame, t0:int, t1:int, prob = True) -> pd.DataFrame:
    deciles = [1,2,3,4,5,6,7,8,9,10]
    result = None
    for decile in deciles:
        selection = data[t1].loc[data[t0] == decile]
        row = np.array([(selection == dec).sum() for dec in deciles])
        if prob and row.sum() >0:
            row = row/row.sum()
        if result is None:
            result = row
        else:
            result = np.vstack((result,row))
    return result

def plot_transition_matrix(matrix: np.ndarray, t0:str, t1:str, ax: matplotlib.axes.Axes = None, show = False) -> None:
    if ax is None:
        fig, ax = plt.subplots( figsize = (5,5), constrained_layout=True, facecolor='ghostwhite')
        fig.suptitle(f"Transition matrix from t{t0} to t{t1}")
        show = True
    ax.imshow(matrix.T, cmap = 'GnBu')
    ax.set_yticks(ticks = np.linspace(start=0,stop=9,num=10), labels = np.linspace(start=1,stop=10,num=10).round(0))
    ax.set_xticks(ticks = np.linspace(start=0,stop=9,num=10), labels = np.linspace(start=1,stop=10,num=10).round(0))
    for row in np.linspace(start=0,stop=9,num=10):
        for col in np.linspace(start=0,stop=9,num=10):
            text = ax.text(row, col, matrix[int(row),int(col)].round(2), ha="center", va="center", color="black")
    ax.set_xlabel(f"t{t0} deciles", fontsize = 14)
    ax.set_ylabel(f"t{t1} deciles", fontsize = 14)
    if show:
        fig.show()
    return


ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
#working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/DynamicExplanation/'
data = pd.read_csv('/Users/nicolasbarticevic/Desktop/simulationOutputs/DynamicExplanation/h_0.csv')
selection = 'h_0'
distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                ENGINE_PATH=ENGINE_PATH,
                                varsigma=100, selectionName=selection, minH=0, maxH=1000,
                                OBS_PERIOD=1, oederByWindow=10,
                                orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                addParam={'Pi'})
stateVariables= ['H', 'N', 'SimpleE', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise', 'InstExp']

seed = 62920828945
#dd = distances.produce(stateVariables=stateVariables, selectionName='dynamicLine', seeds=[62920828944])[62920828944]
dd = distances.produce(stateVariables=stateVariables, selectionName=selection, seeds=[seed])[seed]
#                          tweak={'W':1})[int(seedWithMaxFitnes.iloc[0])]
params = p.printableParams(distances.get_seedParams(seed = seed, selectionName=selection, filterParams= False),
                           paramsPerLine=8, policy=True)


decile_frame = build_deciles_frame(data=dd['H'], method='linear')
fig, ax = plt.subplots(nrows = 1, ncols=4, figsize = (20,5), constrained_layout=True, facecolor='ghostwhite')
t0,t1=90,100
windows = [(0,10), (20,30), (40,50), (60,70)]
for w in range(len(windows)):
    trans = compute_transition(decile_frame, t0 = windows[w][0], t1 = windows[w][1])
    plot_transition_matrix(trans,t0 = windows[w][0], t1 = windows[w][1], ax = ax[w])
fig.show()