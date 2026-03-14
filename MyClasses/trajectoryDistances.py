import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.axes import Axes
from matplotlib.figure import Figure

from MyClasses.pathFinder import PathFinder
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine

from scipy.signal import convolve2d



class TrajectoryDistances(PathFinder):
    def __init__(self, working_directory, allRuns_csv, ENGINE_PATH, varsigma, selectionName, minH, maxH,OBS_PERIOD,
                 oederByWindow, orderByVariable, scaling = "simple", matrixType = "distance", color=0.6, grain = 30,
                 norms = {}, addParam = {}, start = True):
        super().__init__(working_directory=working_directory, allRuns_csv=allRuns_csv, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma,
                         OBS_PERIOD = OBS_PERIOD)
        # Normalizing constants acording to "Sensitivity Analysis model 6.pdf"
        #self.orderedParams = ['N', 'W',  'fixed_capN', 'fixed_delta','fixed_lambda', 'fixed_tau', 'fixed_rho', 'fixed_eta', 'fixed_kappa', 'fixed_capE',
         #'fixed_psi', 'totalCapacity']
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
        if len(addParam) > 0:
            self.addParam(addParam)
        if len(norms) > 0:
            self.repopulateNorms(norms)

        self.selectionName = selectionName
        self.oederByWindow = oederByWindow
        self.orderByVariable = orderByVariable
        self.createSelection(name=self.selectionName, minH=minH, maxH=maxH, color=color)
        self.grain = grain
        self.sorted_seed_value_tuples = self.orderSelection()
        self.scaling = scaling
        self.vectors = self.orderedVectors()
        self.kernelMatrixCells = None
        self.set_matrixType(matrixType)

    def addParam(self, dicParams):
        for param in dicParams:
            self.orderedParams.append(param)

    def repopulateNorms(self, norms):
        for key, value in norms.items():
            setattr(self, f'norm_{key}', value)

    def set_scaling(self, scaling):
        self.scaling = scaling
        self.vectors = self.orderedVectors()

    def set_grain(self, grain):
        self.grain = grain

    def set_matrixType(self, matrixType):
        self.matrixType = matrixType
        if matrixType == "distance":
            self.scaling = "simple"
        if matrixType == "cosine":
            self.scaling = "cosine"

    def orderSelection(self):
        """Order the lines in the selection according to a stateVariable in a window"""
        all_lines = self.produce([self.orderByVariable], selectionName=self.selectionName)
        oneSeed = list(all_lines.keys())[0]
        windowIndex = str(all_lines[oneSeed]["windows"].loc[all_lines[oneSeed]["windows"]['0'] == int(self.oederByWindow)].index[0])
        seed_value_tuples = []
        for seed in all_lines:
            seed_value_tuples.append((seed, all_lines[seed]['H'].loc[:,windowIndex].mean()))
        int_sorted_seed_value_tuples = sorted(seed_value_tuples, key=lambda line: line[1]) #sorts by value
        return int_sorted_seed_value_tuples

    def scaleVector(self, seed):
        """Produce a scaled vector.
        OUTPUT: rows are vectors, columns are dimentions
        """
        ##TODO rewrite the code with only one p in self.orderedParams loop, and switch statementes or whatever they are in python
        params = self.get_seedParams(seed, selectionName=self.selectionName, filterParams = False)
        self.convertStringParams(params)
        result = np.empty(0)
        if self.scaling == 'none':
            for p in self.orderedParams:
                result = np.append(result, params[p])
        if self.scaling == 'simple':
            for p in self.orderedParams:
                result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        if self.scaling == 'capacity':
            for p in self.orderedParams:
                if p == 'totalCapacity' or p == 'N': # Skips N
                    if p == 'totalCapacity':
                        #result = np.append(result, params['totalCapacity']/(5000*params['N']))
                        r = params['totalCapacity']/(params['N'])
                        if r > 1:
                            r = 1.2
                        result = np.append(result, r)
                else:
                    result = np.append(result, params[p]/getattr(self, f'norm_{p}'))
        if self.scaling == 'W_scaled':
            for p in self.orderedParams:
                if p == 'W' or p == 'N': # Skips N
                    if p == 'W':
                        #result = np.append(result, params['W'] / (50 * params['N']))
                        result = np.append(result, params['W'] / (params['N']))

                else:
                    result = np.append(result, params[p]/getattr(self, f'norm_{p}'))

        if self.scaling == 'p_n':
            for p in self.orderedParams:
                if p == "fixed_rho" or p == "fixed_eta": # Skips eta
                    if p == "fixed_rho":
                        #adjust_eta = max(0.1, params['fixed_eta'])
                        #result = np.append(result, params['fixed_rho'] / (100 * adjust_eta))
                        result = np.append(result, params['fixed_rho'] / ( params['fixed_eta']))

                else:
                    result = np.append(result, params[p]/getattr(self, f'norm_{p}'))

        if self.scaling == 'trim_p_n':
            for p in self.orderedParams:
                if p == "fixed_rho" or p == "fixed_eta":
                    if p == "fixed_rho":
                        adjusted_rho = params['fixed_rho']
                        if params['fixed_rho'] > params['fixed_capE']:
                            adjusted_rho = params['fixed_capE']
                        result = np.append(result, adjusted_rho/self.norm_fixed_rho)
                    if p == "fixed_eta":
                        adjusted_eta = params['fixed_eta']
                        if params['fixed_eta'] > params['fixed_capE']:
                            adjusted_eta = params['fixed_capE']
                        result = np.append(result, adjusted_eta/self.norm_fixed_eta)
                else:
                    result = np.append(result, params[p]/getattr(self, f'norm_{p}'))

        if self.scaling == 'multiScaling':
            for p in self.orderedParams:
                if p == 'totalCapacity' or p == 'N' or p == "W" or p == "fixed_rho" or p == "fixed_eta": #skip N and eta
                    if p == 'totalCapacity':
                        r = params['totalCapacity'] / params['N']
                        if r > 1:
                            r = 1.2
                        result = np.append(result, r)
                    if p == 'W':
                        r = params['W'] / params['N']
                        if r > 1:
                            r = 1.2
                        result = np.append(result, r)
                    if p == "fixed_rho":
                        adjusted_rho = params['fixed_rho']
                        if params['fixed_rho'] > params['fixed_capE']:
                            adjusted_rho = params['fixed_capE']
                        adjusted_eta = params['fixed_eta']
                        if params['fixed_eta'] > params['fixed_capE']:
                            adjusted_eta = params['fixed_capE']
                        r = adjusted_rho / adjusted_eta
                        if r > 1:
                            r = 1.2
                        result = np.append(result, r)
                else:
                    result = np.append(result, params[p] / getattr(self, f'norm_{p}'))

        return result

    def print_list_params(self):
        """Return list of params to print in the y axes of the param boxPlots"""
        #TODO rewwrite with only one param in self.orderedParams loop
        result = []
        if self.scaling == "simple" or self.scaling == "none" or self.scaling == "cosine":
            return self.orderedParams
        if self.scaling == "capacity":
            for param in self.orderedParams:
                if param == "totalCapacity" or param == "N":
                    if param == "totalCapacity":
                        result.append("scaledCapacity")
                else:
                    result.append(param)
        if self.scaling == "W_scaled":
            for param in self.orderedParams:
                if param == "W" or param == "N":
                    if param == "W":
                        result.append("scaled_w")
                else:
                    result.append(param)
        if self.scaling == "p_n":
            for param in self.orderedParams:
                if param == "fixed_rho" or param == "fixed_eta":
                    if param == "fixed_rho":
                        result.append("rho/eta")
                else:
                    result.append(param)

        if self.scaling == "trim_p_n":
            for param in self.orderedParams:
                if param == "fixed_rho" or param == "fixed_eta":
                    if param == "fixed_rho":
                        result.append("trim rho")
                    if param == "fixed_eta":
                        result.append("trim eta")
                else:
                    result.append(param)

        if self.scaling == "multiScaling":
            for param in self.orderedParams:
                if param == "capacity" or param == "N" or param == 'W' or param == "fixed_rho" or param == "fixed_eta": #skip N and eta
                    if param == "capacity":
                        result.append("scaledCapacity")
                    if param == "W":
                        result.append("scaled_w")
                    if param == "fixed_rho":   # Skips eta
                        result.append("rho/eta")
                else:
                    result.append(param)


        return result

    def cosineMatrix(self):
        """Return a cosing simmilarity matrix"""

        matrix = np.empty((len(self.vectors), len(self.vectors)))
        for a in range(len(self.vectors)):
            for b in range(len(self.vectors)):
                matrix[a,b] = cosine(self.vectors[a], self.vectors[b])
        return matrix

    def orderedVectors(self):
        """Create a row matrix of ascending order vectors"""
        vectors = []
        seeds = self.sorted_seed_value_tuples
        for seed in seeds:
            vectors.append(self.scaleVector(seed[0]))
        return vectors

    def convertStringParams(self, params: dict):
        """Make string params numeric"""
        #For Pi:
        if 'Pi' in params:
            PiConvert = {'basal': 0, 'H_segmented': 1, 'patient_centred': 2, 'risk': 3, 'need': 4, 'risk_need':5}
            params['Pi'] = PiConvert[params['Pi']]

    def orderedDistanceMatrix(self):
        """Create a distance matrix for all the lines in the selction in ascending oreder"""
        #vectors = self.orderedVectors()
        if self.matrixType == "distance":
            orderedDistancesMatrix = distance_matrix(self.vectors, self.vectors, p = 2)
        if self.matrixType == "cosine":
            orderedDistancesMatrix = self.cosineMatrix()
        return orderedDistancesMatrix

    def downSampledMatrix(self):
        """Run a convolution using a grain x grain kernel and retun the averages"""
        original_matrix = self.orderedDistanceMatrix()
        kernel = np.ones((self.grain, self.grain))
        convolved = convolve2d(original_matrix, kernel, mode='valid')
        matrix_downsampled = convolved[::self.grain, ::self.grain] / (self.grain * self.grain)
        return matrix_downsampled

    def plotOrderedMatrix(self):
        """Plot the distance matrix indicating the percentiles"""
        matrix = self.downSampledMatrix()
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        percentiles = [0,25,50,75,100]
        intervals = (matrix.shape[0]-1) /(len(percentiles)-1)
        lineValues = self.percentileIndexes(p=percentiles)
        lineNumbers= np.arange(start=0, stop=matrix.shape[0]-1+0.01, step=intervals)
        ax.set_xticks(ticks= lineNumbers, labels = lineValues, fontsize = 8)
        ax.set_yticks(ticks= lineNumbers, labels = lineValues, fontsize = 8)
        fig.suptitle(f'Distance matrix for the {self.selectionName} lines. Kernel: {self.grain}x{self.grain}. \n Scaling {self.scaling}.')
        ax.set_xlabel('Quartiles of H at timestep 200', fontsize=12)
        ax.set_ylabel('Quartiles of H at timestep 200', fontsize=12)
        self.kernelMatrixCells = matrix.shape[0]
        print(f'Matrix cells: {self.kernelMatrixCells} ')
        return fig, ax

    def percentileIndexes(self, p=[0,25,50,75,100]):
        """Find the percentiles in the ordered matix"""
        sorted_values = np.array(self.sorted_seed_value_tuples)[:,1]
        lineNumbers = []
        lineValues = []
        for percentile in p:
            value = np.percentile(sorted_values, q=[percentile], method='nearest', axis=0)
            lineNumbers.append(np.argwhere(sorted_values == value)[0][0])
            lineValues.append(round(value[0],2))
        return lineValues

    def scatterParamsForOneLine(self, line, ax):
        """Plot the prarms profile for one line"""
        profile = pd.Series(line, index = self.orderedParams)
        ax.scatter(profile, profile.index.to_list(), alpha = 0.3, color = 'blue')

    def plotParamProfiles(self, lineNumbers: object) -> None:
        """Plot all the profile lines"""
        fig, ax = plt.subplots()
        for number in lineNumbers:
            self.scatterParamsForOneLine(self.vectors[number], ax)
        fig.show()

    def boxParameProfiles(self, cellNumber, ax = None, fig = None):
        """Box plot of parameter profiles for a cell in the plotted downsampled matrix"""
        from_line = cellNumber*self.grain
        to_line = (cellNumber+1)*(self.grain)-1
        lines = np.array(self.vectors)[from_line:to_line,:]
        if ax is None and fig is None:
            fig, ax = plt.subplots()
        ax.boxplot(lines, orientation="horizontal", tick_labels = self.print_list_params())
        ax.set_title(f'cell {cellNumber}')
        #fig.suptitle(f'Parameter profile for cell {cellNumber}')
        return(fig, ax)

    def specialPlotAndSave(self, root):
        """Plot and save the following: (not saving now)
        1. The granular ordered matrix
        2. The 30x30 kernel ordered matrix
        3. A 3-grid boxplot of params for the cell 0, medium and last in the 30x30 kernel matrix"""

        #1
        originalGrain = self.grain
        self.set_grain(1)
        fig, ax = self.plotOrderedMatrix()
        fig.show()
        #fig.savefig(f'{root}/{self.selectionName}ScaledMatrix{self.grain}x{self.grain}_{self.scaling}.png')
        #2
        self.set_grain(30)
        fig, ax = self.plotOrderedMatrix()
        fig.show()
        #fig.savefig(f'{root}/{self.selectionName}ScaledMatrix{self.grain}x{self.grain}_{self.scaling}.png')
        #3
        figure, axes = plt.subplots(ncols=3, nrows=1, figsize=(17, 10))
        fig, axes[0] = self.boxParameProfiles(cellNumber=0, ax=axes[0], fig=figure)
        fig, axes[1] = self.boxParameProfiles(cellNumber=int(self.kernelMatrixCells/2), ax=axes[1], fig=figure)
        axes[1].set_yticks([])
        fig, axes[2] = self.boxParameProfiles(cellNumber=self.kernelMatrixCells-1 , ax=axes[2], fig=figure)
        axes[2].set_yticks([])
        figure.suptitle(f"Patameter ranges for three matix-cells in the {self.grain}x{self.grain} kernel matrix. Scaling {self.scaling}")
        figure.show()
        #figure.savefig(
        #    f'/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/paramBoxPlots{self.selectionName}ScaledMatrix{self.grain}x{self.grain}_{self.scaling}.png')

    def distanceProfile(self):
        """Plot the total distance of every vector to all neigbour lines"""

        matrix = self.orderedDistanceMatrix()
        results = []
        for row in matrix:
            results.append(row.sum())
        fig, ax = plt.subplots(figsize=(30, 10))
        percentiles = [0,25,50,75,100]
        intervals = (matrix.shape[0]-1) /(len(percentiles)-1)
        lineValues = self.percentileIndexes(p=percentiles)
        lineNumbers= np.arange(start=0, stop=matrix.shape[0]-1+0.01, step=intervals)
        ax.plot(results)
        ax.set_xticks(ticks=lineNumbers, labels=lineValues, fontsize=8)
        ax.set_title("Total distance of every parameter vector with all other vectors")
        ax.set_xlabel("Ordered lines (H200-quantiles)")
        ax.set_ylabel("Total distance")
        fig.show()
        return results




# working_directory= f"/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/gridData/500"
# ENGINE_PATH= "/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/CareEngine6_socket.jar"
# #allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/provisionalAllRuns.csv")
# allRuns_csv = pd.read_csv("/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder/runs28_8_25_1.csv")
# scaling = 'simple'
# grain = 30
# selectionName = 'five'
# distances = TrajectoryDistances(working_directory=working_directory, ENGINE_PATH=ENGINE_PATH, allRuns_csv=allRuns_csv,varsigma = 500,
#                         selectionName = selectionName, minH=4, maxH = 5, grain = grain, scaling=scaling)
#
# saveFigTo = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/data/pathFinder'
#
# fig, axe, percentile = distances.plot_5_lines(selection = selectionName ,OBS_PERIOD = 100, window=200,stateVar="H", q = np.arange(0,101, step=10))
# fig.show()
# distances.set_matrixType('distance')
# distances.set_scaling(scaling='simple')
# distances.specialPlotAndSave(root=saveFigTo)
# # distances.set_scaling(scaling='capacity')
# # distances.specialPlotAndSave(root=saveFigTo)
# # distances.set_scaling(scaling = 'W_scaled')
# # distances.specialPlotAndSave(root=saveFigTo)
# # distances.set_scaling(scaling = 'p_n')
# # distances.specialPlotAndSave(root=saveFigTo)
# # distances.set_scaling(scaling = 'trim_p_n')
# # distances.specialPlotAndSave(root=saveFigTo)
# # #
# # distances.set_matrixType('cosine')
# # distances.specialPlotAndSave(root=saveFigTo)
# #
# distances.set_scaling(scaling = 'multiScaling')
# distances.specialPlotAndSave(root=saveFigTo)
#
#
