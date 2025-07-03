from functions.myFunctions import *
from MyClasses.plotting import *
from MyClasses.metrics import *
import matplotlib as mpl


def plot_histograms_visits(c):

    par = {"capacity": [30], "DISEASE_SEVERITY": [1], "LEARNING_RATE": [0],
           "SUBJECTIVE_INITIATIVE": [0.5],
           "numPatients": [1000], "weeks": [50]}
    gridParams = gridCombined(params = par)
    data = c.socket_with_model_paramGrid_2(gridParams)
    metrics = Metrics_socket(data = data, paramGrid= gridParams, descriptions=False)
    Cs= np.array(metrics.data[0]['C']).sum(1)
    C = np.transpose(np.reshape(Cs, (10, 100)))

    cmap = mpl.colormaps['hot_r']
    colors = cmap(np.arange(start= 0.3, stop= 1, step= 0.7/10))
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize = (10, 10), facecolor='ghostwhite')
    axe.hist(C, alpha=0.6, histtype='bar', color = colors)
    fig.show()

    # for i in range(1,11):
    #     print(i)
    #     axe.hist(Cs[Ds == i], alpha = 0.6, histtype='bar')
    fig.show()
    capacity = par["capacity"][0] * par["weeks"][0]
    visits = C.sum()
    print(f'Used capacity: { visits / capacity}')
    print(C.mean(0))
    Ds = np.arange(1,11,1)
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize = (10, 10), facecolor='ghostwhite')
    axe.bar(Ds, height= C.mean(0))
    fig.show()
    print('done')
    return(None)