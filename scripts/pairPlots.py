from functions.myFunctions import *
from MyClasses.plotting import *
from MyClasses.errors import *
import time


def plot_grid_with_vars(dataObject,plot_par_name,grid_par_name, gridParameters, metric_name, filename, folder, errors = False,
                        xgr = 3, ygr = 3):
    p = Plotter()
    fig1, axes = p.plot_grid(dataObject=dataObject, plot_par_name=plot_par_name, grid_par_name=grid_par_name,
                           metric_name=metric_name, grid=gridParameters, xgrids=xgr, ygrids=ygr,
                           variability=True, errors = errors, ymax=125)
    fig1.show()
    fig1.savefig(f"{folder}{filename}.png")
    return fig1, axes

def plot_vars(dataObject,plot_par_name, grid_par_name, metric_name, filename, folder, gridParameters):
    p = Plotter()
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10,10), facecolor='ghostwhite')
    p.one_plot2(dataObject = dataObject, axe = axe, plot_par_name = plot_par_name, grid_par_name = grid_par_name,
                    metric_name = metric_name,  sub_grid = gridParameters, variability = True, labels=['1 disease', '2 diseases', '3 diseases'])
    fig.show()
    fig.savefig(f"{folder}{filename}.png")
    return fig

def configParams(par1 = "", par2 = ""):
    """ Build the grid of parameters for pairs."""
    par = ensamble_par(par1, par2)
    print(par)
    return gridCombined(par)

def ensamble_par(par1 = "", par2 = ""):
    """Build the param dictionary for pairs."""
    a = np.arange(start=0, stop=1.1, step=.13);a.put(8,1)
    par = {"capacity": [80], "DISEASE_SEVERITY": [3], "LEARNING_RATE": [0],
           "SUBJECTIVE_INITIATIVE": [0.5],
           "numPatients": [1000], "weeks": [150]}
    if par1 == 'capacity':
        par[par1] = np.arange(start=0, stop=700, step=80)
    if par2 == 'capacity':
        par[par2] = np.arange(start=0, stop=700, step=80)
    if par1 == 'DISEASE_SEVERITY':
        par[par1] = np.arange(start=0, stop=10, step=1.2)
    if par2 == 'DISEASE_SEVERITY':
        par[par2] = np.arange(start=0, stop=10, step=1.2)
    if par1 == 'LEARNING_RATE' or par1 == 'SUBJECTIVE_INITIATIVE':
        par[par1] = a
    if par2 == 'LEARNING_RATE' or par2 == 'SUBJECTIVE_INITIATIVE':
        par[par2] = a
    if isinstance(par1, dict):
        par[next(iter(par1))] = par1[next(iter(par1))]
    if isinstance(par2, dict):
        par[next(iter(par2))] = par2[next(iter(par2))]
    return par

delay_for_socket = 1

def do_pairs(c,all_pairs = {
    #0: ['capacity', 'DISEASE_SEVERITY', 'pairs_capacity_disease_1000patients'],
    #1: ['capacity', 'LEARNING_RATE', 'pairs_capacity_learning_1000patients'],
    #2: ['capacity', 'SUBJECTIVE_INITIATIVE', 'pairs_capacity_subjective_1000patients'],
    #3: ['DISEASE_SEVERITY', 'LEARNING_RATE', 'pairs_disease_learning_1000patients'],
    #4: ['DISEASE_SEVERITY', 'SUBJECTIVE_INITIATIVE', 'pairs_disease_subjective_1000patients'],
    #5: ['LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', 'pairs_learning_subjective_1000patients'],
    6: ['SUBJECTIVE_INITIATIVE', 'capacity', 'pairs_subjective_capacity_1000patients' ]}):

    folder = f'./figures/{c.get_model_name()}/participation/pairs/'
    for i in all_pairs:
        c.start_server()
        time.sleep(delay_for_socket)
        gridParameters = configParams(all_pairs[i][0], all_pairs[i][1])
        data = c.socket_with_model_paramGrid_2(gridParameters)
        metrics = Metrics_socket(data, gridParameters, False)
        metrics.configureVariability('sd')
        plot_grid_with_vars(dataObject=metrics,plot_par_name= get_plot_par_name(all_pairs[i]),  grid_par_name=all_pairs[i][1], gridParameters=gridParameters,
                   metric_name = "attempts_patient_simulation", filename = f'{all_pairs[i][2]}_var', folder=folder,xgr=3, ygr=3)
        c.start_server()
        time.sleep(delay_for_socket)
        errors = Errors_socket(gridParameters=gridParameters, N=10, client = c)
        plot_grid_with_vars(dataObject=errors,plot_par_name= get_plot_par_name(all_pairs[i]),  grid_par_name=all_pairs[i][1], gridParameters=gridParameters,
                        metric_name="attempts_patient_simulation", filename=all_pairs[i][2],
                        folder=folder, errors=True, xgr=3, ygr=3)
    return None

def get_plot_par_name(list):
    if isinstance(list[0], dict):
        return next(iter(list[0]))
    else:
        return list[0]