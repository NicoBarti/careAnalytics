import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from MyClasses.pathFinderCollector import PathFinderCollector
#from MyClasses.pathFinder import PathFinder
#from MyClasses.trajectoryDistances import TrajectoryDistances
#from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
import seaborn as sns
from MyClasses.lineGroup import LineGroup
from MyClasses.trajectoryDistances import TrajectoryDistances
import time

class ParamPlots():
    """Process a dataframe of params and results"""

    def __init__(self, data, excluded = [], scaling = "simple"):
        # Normalizing constants acording to "Sensitivity Analysis model 6.pdf"
        #self.orderedParams = ['N', 'W',  'fixed_capN', 'fixed_delta','fixed_lambda', 'fixed_tau', 'fixed_rho', 'fixed_eta', 'fixed_kappa', 'fixed_capE',
         #'fixed_psi', 'totalCapacity']
        self.data = data
        self.excluded = excluded

        self.orderedParams = [  'fixed_lambda', 'fixed_tau',  'fixed_kappa', 'fixed_rho', 'fixed_eta','fixed_capN','fixed_capE',
         'fixed_psi', 'N', 'W', 'fixed_delta','totalCapacity']
        self.norm_N = 5000
        self.norm_W = 50
        self.norm_fixed_delta = 10
        self.norm_fixed_capN = 10
        self.norm_fixed_lambda = 10
        self.norm_fixed_tau = 10
        self.norm_fixed_rho = 10
        self.norm_fixed_eta = 10
        self.norm_fixed_kappa = 1
        self.norm_fixed_capE = 10
        self.norm_fixed_psi = 1
        self.norm_totalCapacity = 5000
        self.scaling = scaling
        self.vectors = self.generateVectors()
        self.kernelMatrixCells = None

    def set_scaling(self, scaling):
        self.scaling = scaling
        self.vectors = self.generateVectors()

    def generateVectors(self):
        paramData = self.data.loc[:,('fixed_capN', 'fixed_kappa', 'fixed_delta', 'fixed_psi',
         'N', 'fixed_lambda', 'fixed_tau', 'fixed_eta', 'totalCapacity', 'W', 'Pi', 'fixed_capE', 'fixed_rho')]
        return paramData.drop(columns = self.excluded, errors='ignore')

    def excludeParams(self,values: list):
        self.excluded = values
        self.vectors = self.generateVectors()

    def scaleVectors(self):
        """Produce a scaled vector.
        OUTPUT: rows are vectors, columns are dimentions
        """

        result = np.empty(0)
        paramList = [x for x in  self.orderedParams if x not in self.excluded]
        result = pd.DataFrame()
        for param in paramList:
            result = pd.concat([result, self.vectors.loc[:, param] / getattr(self, f'norm_{param}')], axis=1)
        return result

        # if self.scaling == 'capacity':
        #     for p in self.orderedParams:
        #         if p == 'totalCapacity' or p == 'N': # Skips N
        #             if p == 'totalCapacity':
        #                 #result = np.append(result, params['totalCapacity']/(5000*params['N']))
        #                 r = params['totalCapacity']/(params['N'])
        #                 if r > 1:
        #                     r = 1.2
        #                 result = np.append(result, r)
        #         else:
        #             result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        # if self.scaling == 'W_scaled':
        #     for p in self.orderedParams:
        #         if p == 'W' or p == 'N': # Skips N
        #             if p == 'W':
        #                 #result = np.append(result, params['W'] / (50 * params['N']))
        #                 result = np.append(result, params['W'] / (params['N']))
        #
        #         else:
        #             result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        #
        # if self.scaling == 'p_n':
        #     for p in self.orderedParams:
        #         if p == "fixed_rho" or p == "fixed_eta": # Skips eta
        #             if p == "fixed_rho":
        #                 #adjust_eta = max(0.1, params['fixed_eta'])
        #                 #result = np.append(result, params['fixed_rho'] / (100 * adjust_eta))
        #                 result = np.append(result, params['fixed_rho'] / ( params['fixed_eta']))
        #
        #         else:
        #             result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        #
        # if self.scaling == 'trim_p_n':
        #     for p in self.orderedParams:
        #         if p == "fixed_rho" or p == "fixed_eta":
        #             if p == "fixed_rho":
        #                 adjusted_rho = params['fixed_rho']
        #                 if params['fixed_rho'] > params['fixed_capE']:
        #                     adjusted_rho = params['fixed_capE']
        #                 result = np.append(result, adjusted_rho/self.norm_fixed_rho)
        #             if p == "fixed_eta":
        #                 adjusted_eta = params['fixed_eta']
        #                 if params['fixed_eta'] > params['fixed_capE']:
        #                     adjusted_eta = params['fixed_capE']
        #                 result = np.append(result, adjusted_eta/self.norm_fixed_eta)
        #         else:
        #             result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        #
        # if self.scaling == 'multiScaling':
        #     for p in self.orderedParams:
        #         if p == 'totalCapacity' or p == 'N' or p == "W" or p == "fixed_rho" or p == "fixed_eta": #skip N and eta
        #             if p == 'totalCapacity':
        #                 r = params['totalCapacity'] / params['N']
        #                 if r > 1:
        #                     r = 1.2
        #                 result = np.append(result, r)
        #             if p == 'W':
        #                 r = params['W'] / params['N']
        #                 if r > 1:
        #                     r = 1.2
        #                 result = np.append(result, r)
        #             if p == "fixed_rho":
        #                 adjusted_rho = params['fixed_rho']
        #                 if params['fixed_rho'] > params['fixed_capE']:
        #                     adjusted_rho = params['fixed_capE']
        #                 adjusted_eta = params['fixed_eta']
        #                 if params['fixed_eta'] > params['fixed_capE']:
        #                     adjusted_eta = params['fixed_capE']
        #                 r = adjusted_rho / adjusted_eta
        #                 if r > 1:
        #                     r = 1.2
        #                 result = np.append(result, r)
        #         else:
        #             result = np.append(result, params[p] / getattr(self, f'norm_{p}'))


