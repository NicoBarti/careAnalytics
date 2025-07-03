import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

class Plotter:
    def __init__(self, cmap = mpl.colormaps['cividis'], c = [2 / 3 + 1 / 6, 1 / 3 + 1 / 6, 1 / 6 + 0.07],
                 labels = ['1 disease', '2 diseases', '3 diseases']):
        self.cmap = cmap
        self.c = self.cmap(c)
        self.labels = labels
        #self.linestyle = 'solid'
        self.linestyle = ['dotted', 'dashed', 'solid']
        #local_path = "/Users/nicolasbarticevic/"
        #output_fig_path = local_path + "Desktop/latex/figures/"
        #rng = np.random.default_rng()

    def set_labels(self, bol):
        if isinstance(bol, bool):
            if bol:
                self.labels =['1 disease', '2 diseases', '3 diseases']
            else:
                self.labels = [None,None,None]
        else:
            print("Must pass a boolean to set_labels")
            return()

    def set_colors(self, c):
        if c == "diseasesNumber":
            self.cmap = mpl.colormaps['cividis']
            self.c = self.cmap([2 / 3 + 1 / 6, 1 / 3 + 1 / 6, 1 / 6 + 0.07])
            self.linestyle = ['solid']*3
        if c == "expectations":
            self.cmap = mpl.colormaps['PiYG']
            self.c = self.cmap([0.9,0.9,0.9])
            self.linestyle = ['dashed']*3
        if c == "needs":
            self.cmap = mpl.colormaps['PiYG']
            self.c = self.cmap([0.1,0.1,0.1])
            self.linestyle = ['dotted']*3
        if c == "visits":
            self.cmap = mpl.colormaps['viridis']
            self.c = self.cmap([0.4,0.4,0.4])
            self.linestyle = ['dashed']*3
        if c == "attempts":
            self.cmap = mpl.colormaps['viridis']
            self.c = self.cmap([0.6,0.6,0.6])
            self.linestyle = ['dotted']*3
        if c == "treatments":
            self.cmap = mpl.colormaps['viridis']
            self.c = self.cmap([0.7,0.7,0.7])
            self.linestyle = ['dashdot']*3
        if c == "progression":
            self.cmap = mpl.colormaps['viridis']
            self.c = self.cmap([0.8,0.8,0.8])
            self.linestyle = [(0, (1, 1))]*3 #densly dotted
        if c == "black":
            self.c = ['black', 'black', 'black']
            self.linestyle = ['solid']*3

        if c not in ['black', 'attempts', 'visits', 'needs', 'expectations', 'diseasesNumber','treatments']:
            print("Must pass one of the following strings to set_colors:")
            print(['black', 'attempts', 'visits', 'needs', 'expectations', 'diseasesNumber','treatments'])

    def get_colors(self):
        return(self.c)

    def get_linestyle(self):
        return(self.linestyle)

    def plot_grid(self,dataObject, plot_par_name, grid_par_name, metric_name, grid,
                  ygrids="", xgrids="", size=6, sup_title="",
                  errors=False, variability=False, ymax=None, ymin=None):
        ##compute ygrids and ygrids if not given
        if ygrids == "" and xgrids == "":
            half = len(grid[grid_par_name].unique()) / 2
            ygrids = int(half)
            xgrids = int(half) + len(grid[grid_par_name].unique()) % half
        ##Wrapper function to make grid plots
        fig, axes = plt.subplots(ygrids, xgrids, figsize=(size * xgrids, size * ygrids), facecolor='ghostwhite')
        xcord, ycord = 0, 0
        maximum_ys = []
        minimum_ys = []
        for i in grid[grid_par_name].unique():
            if ygrids == 1 & xgrids == 1:
                axe = axes
            elif ygrids == 1:
                axe = axes[xcord]
            else:
                axe = axes[ycord, xcord]
            ##Call the plotting function
            if errors:
                p = self.one_error_plot2(dataObject=dataObject, axe=axe, plot_par_name=plot_par_name,
                                    grid_par_name=grid_par_name, metric_name=metric_name,
                                    sub_grid=grid.loc[grid[grid_par_name] == i],
                                     bars= variability)
            else:
                p = self.one_plot2(dataObject=dataObject, axe=axe, plot_par_name=plot_par_name, grid_par_name=grid_par_name,
                              metric_name=metric_name, sub_grid=grid.loc[grid[grid_par_name] == i],
                               variability=variability)

            ## get the max yview point to homogenize the y axes later
            maximum_ys.append(p.viewLim.get_points()[1][1])
            minimum_ys.append(p.viewLim.get_points()[0][1])

            if (xcord == xgrids - 1):
                xcord = 0
                ycord = ycord + 1
            else:
                xcord = xcord + 1
        ## reset y_lims to homogenize output across the gird
        if ymax is not None:
            new_y_max = ymax
        else:
            new_y_max = np.array(maximum_ys).max()
        if ymin is not None:
            new_y_min = ymin
        else:
            new_y_min = np.array(minimum_ys).min()
        for y in range(0, ygrids):
            for x in range(0, xgrids):
                axes[y, x].set_ylim(top=new_y_max, bottom=new_y_min)
        return (fig, axes)


    def one_error_plot2(self, dataObject, axe, plot_par_name, grid_par_name, metric_name, sub_grid, ds=[1, 2, 3], bars=True,
                        colideGroups = False):
        ## Makes one plot. Mean to be used in a grid generator
        ##INPUTS:
        # plot_par_name the variable name for the x axis
        # grid_par_name the variable for the title
        # metric_name the metric to call from the Metric class
        # sub_grid is the data. MUST have one column = grid_par_name whit only one value

        # Check that the sub_grid is ok
        grid_par_values = sub_grid[grid_par_name].unique()
        if (len(grid_par_values) != 1):
            print(f"in function one_plot, the given sub_grid contains several values for {grid_par_name}")
            print(f"grid_par_name {sub_grid[grid_par_name].unique()}")
            print("Should contain only one. ABORTING PLOT")
            return ()
        grid_par_value = sub_grid[grid_par_name].unique()[0]

        bootstrapedMetrics = dataObject.bootstrapMetric(metric_name=metric_name, plot_par_name=plot_par_name,
                                                        grid_param_name= grid_par_name, colideGroups=colideGroups)
        axe.set_xlabel(plot_par_name, size=12)
        axe.set_ylabel(dataObject.dic['metrics0'].call(metric_name), size=12)
        for i in range(0, 3):
            if i + 1 in ds:
                mean = bootstrapedMetrics[grid_par_value][f'd{i + 1}']['mean']
                xs = sub_grid[plot_par_name].loc[sub_grid[grid_par_name] == grid_par_value]

                if bars:
                    asymetric_error = [mean - bootstrapedMetrics[grid_par_value][f'd{i + 1}']['lower'],
                                   bootstrapedMetrics[grid_par_value][f'd{i + 1}']['high'] - mean]
                    axe.errorbar(xs, mean, yerr=asymetric_error, color=self.c[i], label=self.labels[i], elinewidth=2, linewidth=1,
                                 linestyle=self.linestyle[i])
                else:
                    axe.plot(xs, mean, color=self.c[i], label=self.labels[i])
        if self.labels != [None,None,None]:
            axe.legend(prop={'size': 10})
        axe.set_title(f'{grid_par_name}= {grid_par_value}')
        axe.set_xticks(xs)
        return (axe)


    def one_plot2(self,dataObject, axe, plot_par_name, grid_par_name, metric_name, sub_grid, ds=[1, 2, 3],
                  variability=False):
        ## Makes one plot. Mean to be used in a grid generator
        ##INPUTS:
        # plot_par_name the variable name for the x axis
        # grid_par_name the variable for the title
        # metric_name the metric to call from the Metric class
        # sub_grid is the data. MUST have one column = grid_par_name whit only one value

        # Check that the sub_grid is ok
        grid_par_values = sub_grid[grid_par_name].unique()
        if (len(grid_par_values) != 1):
            print(f"in function one_plot, the given sub_grid contains several values for {grid_par_name}")
            print(f"grid_par_name {sub_grid[grid_par_name].unique()}")
            print("Should contain only one. ABORTING PLOT")
            return ()
        # array for the metrics
        e = np.empty((0, 3))
        for i in sub_grid.index:
            e = np.append(e, [dataObject.call(metric_name, i, plot_par_name, grid_par_name)], axis=0)
        if variability:
            v = np.empty(((0, 3, 2)))  # (plot_par_value, seveiry, lower-upper)
            for i in sub_grid.index:
                v = np.append(v, [dataObject.call(metric_name, i, plot_par_name, grid_par_name, variability=True)], axis=0)
        ##Plot

        for i in range(0, 3):
            if i + 1 in ds:
                xs = sub_grid[plot_par_name]
                #if variability:
                #    xs = xs + rng.standard_normal(size=len(xs)) / 2 //this was adding noise, but in some plots it was out of scale
                if variability and dataObject.getVariability() == 'quantile':
                    lower = e[:, i] - v[:, i, 0]
                    upper = v[:, i, 1] - e[:, i]
                    np.place(lower, lower < 0,
                             0)  ##as I'm mixing means with quantiles, sometimes I get ngative erro bar sizes. Put 0 in these cases
                    np.place(upper, upper < 0, 0)
                    axe.errorbar(xs, e[:, i], yerr=[lower, upper], color=self.c[i], label=self.labels[i], elinewidth=2, linewidth=0.7, linestyle = self.linestyle)
                    axe.scatter(xs, e[:, i], color=self.c[i])
                    errortitle = " [q25-75]"
                elif variability and dataObject.getVariability() == 'sd':
                    axe.errorbar(xs, e[:, i], yerr=v[:, i, 0], color=self.c[i], label=self.labels[i], elinewidth=2, linewidth=0.7, linestyle = self.linestyle)
                    axe.scatter(xs, e[:, i], color=self.c[i])
                    errortitle = " [±sd]"
                else:
                    axe.scatter(xs, e[:, i], color=self.c[i], label=self.labels[i])
                    axe.plot(xs, e[:, i], color=self.c[i])
                    errortitle = ""
        axe.legend(prop={'size': 10})
        axe.set_title(f'{grid_par_name}= {grid_par_values[0]}')
        axe.set_xlabel(plot_par_name, size=12)
        labely = dataObject.call(metric_name)
        axe.set_ylabel(labely + errortitle, size=12)
        axe.set_xticks(sub_grid[plot_par_name])
        return (axe)

    def one_needexpe_hist(self, dataObject, axe, plot_par_name, grid_par_name, sub_grid, type, plot_par_value,ds=[1, 2, 3]
                          ):
        # def multiHist(data, axe, weeks, c, colName="C", bins=5, title="", Legend=True):
        #     # bins = np.array([0,1,2,3,4,5])
        bins = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        p = Plotter()
        if type == 'needs':
            p.set_colors('needs')
            callName = "last_step_needs"
            axe.set_xlabel('Number of needs in the last step',size=12)

        elif type == 'exp':
            p.set_colors('expectations')
            callName = "last_step_exp"
            axe.set_xlabel('Number of expectations in the last step',size=12)
        else:
            return(None)
        row = sub_grid.loc[sub_grid[plot_par_name] == plot_par_value].index[0]
        e = dataObject.call(callName, row, plot_par_name, grid_par_name)
        color = p.get_colors()
        axe.hist(e[0], bins, histtype='bar', color=color[0], align='mid')
        axe.set_xticks(bins)
        axe.set_ylabel('Number of patients',size=12)
        axe.set_title(f'{plot_par_name} {plot_par_value}', size=12)
        return(axe)