#I'll reproduce the line using trajectory
#I'll use linePlotter's to plot the line
from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
import matplotlib.pyplot as plt
import numpy as np


working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/reproduceLines/outcomeMatrix'
ENGINE_PATH = "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar"
params = {'fixed_lambda': 0.5000, 'fixed_tau': 2, 'fixed_kappa': 0.0961, 'fixed_rho': 1.0000, 'fixed_eta': 1.0000,
          'fixed_capN': 10.0000, 'fixed_capE': 10.0000, 'fixed_psi': 0.5000,  'N': 936, 'W': 1, 'fixed_delta': 5.0000,
          'totalCapacity': 60, 'varsigma': 500, 'policy': 'basal'}
# params = {'fixed_lambda': 0.5000 , 'fixed_tau': 1.9928 , 'fixed_kappa': 0.0961 , 'fixed_rho': 1.0000 , 'fixed_eta': 1.0000 ,
#           'fixed_capN': 10.0000 , 'fixed_capE': 10.0000 , 'fixed_psi': 0.5000 , 'N': 2844 , 'W': 1 , 'fixed_delta': 5.0000 ,
#           'totalCapacity': 189 , 'policy': 'basal', 'varsigma': 100}

line = Trajectory(working_directory = working_directory, selectionName="q1", ENGINE_PATH=ENGINE_PATH, params=params)

a = np.array(line.runLine()[0]['H'])[:,1]

plotter = LinePlotter()
fig, axe = plt.subplots(nrows=1, ncols=1, constrained_layout=True, facecolor='ghostwhite', figsize=(20, 7))
axe.hist(a)
fig.show()
print(a.var()/((500*5/52)*(500*5/52)))


