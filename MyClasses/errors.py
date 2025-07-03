from scipy.stats import bootstrap
from MyClasses.metrics import *


class Errors_socket:
    """Call the model many times and compute the error of metrics."""

    ## This class computes the error bars. You should have one of these objects for each new grid. These are the steps:
    ## 1. Call the current model using the passed param grid (combinations of parameters). It calls the model N times per param grid line
    ## 2. It saves these runs so you only have to call them once.
    ## 3. It uses the Metrics class to compute metrics and bootstraps
    ## The output is a dictionary with the metrics and the limits.

    def __init__(self, gridParameters, N, client,reuse_previous_output=False, vervoso=False, method="percentile",
                 fetchType = 'OK_params'):
        self.reuse_previous_output = reuse_previous_output
        self.vervoso = vervoso
        self.gridParameters = gridParameters
        self.method = method
        self.c = client
        # dictionary to store the results of the simulations
        self.dic = {}
        if (N < 2):
            print("N can't be < 2")
            self.N = 2
        else:
            self.N = N
        self.fetchType = fetchType
        self.runEngine()

    def runEngine(self):
        """Run the simulations and store them in dic."""
        frames = self.c.socket_with_model_paramGrid_2(self.gridParameters, ComputeErrors=self.N, fetchType=self.fetchType)
        for i in range(0, self.N):
            self.dic[f'metrics{i}'] = Metrics_socket(frames[i], self.gridParameters, False)

    def bootstrapMetric(self, metric_name, plot_par_name, grid_param_name, colideGroups = False):
        ##Main utility function
        ## empty boostrap dic to store each bootstraps dic per grid-par-value
        bootstraps = {}
        ##per each par_grid_value
        for grid_par_value in self.gridParameters[grid_param_name].unique():
            ## FIRST: assamble the respective subGrid
            sub_grid = self.makesubGrid(grid_param_name, grid_par_value)
            ## SECOND: compute the boostrap and store with key
            bootstraps[grid_par_value] = self.computeBoostrap(sub_grid, metric_name, plot_par_name, grid_param_name, colideGroups)
        ## RETURN the sults
        return (bootstraps)

    def makesubGrid(self, grid_param_name, grid_par_value):
        sub_grid = self.gridParameters.loc[self.gridParameters[grid_param_name] == grid_par_value]
        return (sub_grid)

    ## compute metric and boots for each value of plot_par_name in the sub_grid
    def computeBoostrap(self, sub_grid, metric_name, plot_par_name, grid_param_name, collideGroups = False):
        # Dictionary for the boosttrap
        bootstraped = {'d1': {'mean': np.empty(0), 'lower': np.empty(0), 'high': np.empty(0)},
                       'd2': {'mean': np.empty(0), 'lower': np.empty(0), 'high': np.empty(0)},
                       'd3': {'mean': np.empty(0), 'lower': np.empty(0), 'high': np.empty(0)}}
        ## COMPUTE the metrics:
        for index_number in sub_grid.index:
            ## compute the metric using the data from each simulation ans stores in computed_metric
            computed_metric = []
            for i in range(0, len(self.dic)):
                computed_metric.append(
                    self.dic[f'metrics{i}'].call(method_name=metric_name, paramRowNumber=index_number,
                                                 par1=plot_par_name, par2=grid_param_name, no_group=collideGroups))
            computed_metric = np.array(computed_metric)

            ## COMPUTE the bootstrap
            for i in range(0, 3):
                bootstraped[f'd{i + 1}']['mean'] = np.append(bootstraped[f'd{i + 1}']['mean'],
                                                             computed_metric[:, i].mean())
                confint = bootstrap((computed_metric[:, i],), np.mean, method=self.method).confidence_interval
                bootstraped[f'd{i + 1}']['lower'] = np.append(bootstraped[f'd{i + 1}']['lower'], confint.low)
                bootstraped[f'd{i + 1}']['high'] = np.append(bootstraped[f'd{i + 1}']['high'], confint.high)
        return (bootstraped)

    def getAverageAtt(self, nrow):
        averageB_sim = np.empty((0,3))
        for i in range(0, self.N):
            B = self.dic[f'metrics{i}'].call(method_name="attempts_patient_simulation", paramRowNumber=nrow, par1='capacity', par2='weeks')
            averageB_sim = np.append(averageB_sim, [B], axis=0)
        return(averageB_sim)


