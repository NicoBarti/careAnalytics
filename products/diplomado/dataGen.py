import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.pathFinder import PathFinder
from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
import seaborn as sns
from MyClasses.lineGroup import LineGroup
import pandas as pd
from MyClasses.client import Client

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar'
c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()
varsigma = 500
def run_model(stateVariables, params):
    """Run the model for the seeds in selectionName, observing stateVariables with OBS_PERIOD
    INPUT: stateVariables is a list of str with the var names, OBS_PERIOD is an int, selectionName a str
    OUTPUT: dictionary of seed: {stateVariables: results}"""
    simdata = {}
    params['PROVIDER_INIT'] = ['applyFixed']
    params['PATIENT_INIT'] = ['classExample']
    params["OBS_PERIOD"] = [varsigma]
    params["reproduce_line"] = ["true"]
    for stateVariable in stateVariables:
        params[f"obs{stateVariable}"] = "true"
        run_data = c.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(params))
        c.start_server()
        simdata[stateVariable] = pd.DataFrame(run_data[0][stateVariable], columns=run_data[0]['windows'])
    simdata['delta'] = pd.DataFrame(run_data[0]['delta'], columns=run_data[0]['windows'])
    return (simdata)

stateVariables = ['H', 'N', 'SimpleE', 'SimpleC']

policy = 'H_segmented'
# params = {'Pi' : policy ,
#               'fixed_kappa': 0.2,
#               'varsigma': varsigma,
#               'N': 1000,
#               'fixed_lambda': 1,
#               'fixed_tau': 2,
#               'fixed_eta': 0.5,
#               'totalCapacity': 40,
#               'fixed_capE': 10,
#               'fixed_capN':5,
#               'fixed_psi':0.5,
#               'fixed_rho': 2}
# data = run_model(stateVariables= stateVariables, params=params)
#
# plot_data = pd.DataFrame({'H': data['H'][varsigma], 'N': data['N'][varsigma], 'SimpleE': data['SimpleE'][varsigma],
#                           'SimpleC': data['SimpleC'][varsigma], 'delta': data['delta'][varsigma]})
# plot_data.to_csv(f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/products/diplomado/{policy}.csv')

plot_data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/products/diplomado/{policy}.csv')

sns.histplot(plot_data['H'])
plt.title(policy)
plt.show()
sns.histplot(plot_data['SimpleC'])
plt.title(policy)

plt.show()
sns.histplot(plot_data['SimpleE'])
plt.title(policy)

plt.show()
sns.relplot(x = plot_data['H'], y= plot_data['SimpleE'])
plt.title(policy)

plt.show()
sns.relplot(x = plot_data['H'], y= plot_data['SimpleC'])
plt.title(policy)

plt.show()
sns.relplot(x = plot_data['SimpleE'], y= plot_data['SimpleC'])
plt.title(policy)

plt.show()
print(plot_data['H'].mean(0))
print(0)


data = plot_data.drop(columns='Unnamed: 0')
data.rename(columns = {'H': 'Sintomas', 'N': 'MotivosConsulta', 'SimpleE': 'Satisfaccion_Acceso', 'SimpleC': 'Consultas', 'delta': 'Enfermedades'}, inplace = True)
data = data.drop(columns='MotivosConsulta')
e = data['Enfermedades']
e = e.replace([2,4,6], ['1 dg','2 -3 dg','4 + dg'])
data.drop(columns = 'Enfermedades', inplace = True)
data = pd.DataFrame.join(data,e)
data.to_csv(f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/products/diplomado/data_{policy}.csv', index=False)