import pandas as pd
import numpy as np

class Metrics_socket:
    """Store the simulation data and compute metrics on them."""

    ##It keeps copyes of the matrices used for previous calls to make it fast. Retreiving a specific matrix the second time is 70 times faster this way
    def __init__(self, data, paramGrid, descriptions=True, variability="quantile"):
        self.data = data
        self.paramGrid = paramGrid
        self.configureVariability(variability)
        if (descriptions):
            self.list_available_metrics()

    def call(self, method_name, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        func = getattr(self, method_name)
        return (func(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group=no_group))

    def _extractMatrix(self, name, paramRowNumber, par1, par2):
        """Extract and save the matrices from data."""
        # INPUT
        ##name, the name of the matrix to extract (N, B, C, T, E; see model documnetation)
        ##the paramRowNumber that identify the simulation
        ##OUTPUT
        ## A dataFrame with the matix of interest: with row number of patients, and column number of weeks
        completeMatrix = pd.DataFrame(self.data[paramRowNumber][name])
        mask = pd.Series(self.data[paramRowNumber]['d'])
        setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d1', completeMatrix.loc[mask == 1])
        setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d2', completeMatrix.loc[mask == 2])
        setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d3', completeMatrix.loc[mask == 3])

        ##segment of data
        # subdata = self.data[(self.data[par1] == str(self.paramGrid[par1][paramRowNumber])) & (self.data[par2] == str(self.paramGrid[par2][paramRowNumber]))][mask]
        # subdata = self.data[(self.data[par1] == str(self.paramGrid.loc[paramRowNumber,:][par1])) & (self.data[par2] == str(self.paramGrid.loc[paramRowNumber,:][par2]))][mask]
        # setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d1', subdata[subdata["d"] == 1].drop("d", axis = 1))
        # setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d2', subdata[subdata["d"] == 2].drop("d", axis = 1))
        # setattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d3', subdata[subdata["d"] == 3].drop("d", axis = 1))
        return (getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d1'),
                getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d2'),
                getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d3'))

    def _fetch(self, name, paramRowNumber, par1, par2):
        """Check if you have the data. If do, fetch it. If not, call _extractMatrix()."""
        attributeName = f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d1'
        if (hasattr(self, attributeName)):
            return (getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d1'),
                    getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d2'),
                    getattr(self, f'SubdataRow{paramRowNumber}name{name}{par1}{par2}d3'))
        else:
            return self._extractMatrix(name, paramRowNumber, par1, par2)

    def logcalTestingData(self, paramRowNumber, par1, par2):
        ##See if all decreases in needs are due to visit (treatment) effects
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            diferences = np.diff(Ns[i], axis=1)
            satisfactions = (diferences * (diferences < 0))
            visits_lagged = Cs[i].to_numpy()[:, :-1]  ## the treatment effect is next week
            no_visits_lagged = (visits_lagged - 1) * -1
            ## should get all 0 in this multiplication, because there are no satisfactions with no_visits
            check = satisfactions * no_visits_lagged
            if (check.sum() != 0):
                print("THIS DATA VIOLATES ASSUMTION THAT ONLY TREATMENT MAKES NEEDS DECREASE")
                print("This assumption is used for the metrics computation")
                print(self.paramGrid.loc[paramRowNumber, :])
                print("These needs decreased without treatment:")
                print(f"For stratum {i + 1}")
                print(check)
                return (False)
        return (True)

    def configureVariability(self, function):
        if function != "quantile" and function != "sd":
            print("only quantile and sd available")
        else:
            self.var = function

    def getVariability(self):
        return self.var

    def list_available_metrics(self):
        print("")
        print("ACCESS metrics:")
        print("attempts_patient_simulation \t" + self.attempts_patient_simulation())
        print("ratio_succes_unsuccess \t\t" + self.ratio_succes_unsuccess())
        print("avg_visits \t\t\t" + self.avg_visits())
        print("prop_success \t\t\t" + self.prop_success())
        print("unsuccessful_attempts_patient\t" + self.unsuccessful_attempts_patient())
        print("visits_per_patient\t\t" + self.visits_per_patient())
        print("")
        print("PERFORMANCE metrics:")
        print("ratio_satisfied_visits \t\t" + self.ratio_satisfied_visits())
        print("ratio_visits_satisfied \t\t" + self.ratio_visits_satisfied())
        print("ratio_not_satisfied_visits\t" + self.ratio_not_satisfied_visits())
        print("satisfaction_perVisit_perPatient  " + self.satisfaction_perVisit_perPatient())
        print("need_evolution \t\t\t" + self.need_evolution())
        print("not_satisfied_needs_patient_simulation\t" + self.not_satisfied_needs_patient_simulation())
        print("capacity_usage\t\t\t" + self.capacity_usage())
        print("")
        print("Note: Needs decrease only by the effect of treatments, there are no other ways for satisfaction")
        print("In this senario, the T matrix was not used, as it can be infered from C and H")
        print("")
        print("OTHER:")
        print("mean_expectations \t\t" + self.mean_expectations() + " TESTING PENDING")
        print("mean_needs \t\t" + self.mean_needs() + " TESTING PENDING")
        print("expectation_end_sim"+ self.expectation_end_sim()+"TESTING PENDING")


    ####
    #### Access metrics
    ####

    def attempts_patient_simulation(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ## OUTPUT:
        ## Average number of attempts on the whole simulation
        if (paramRowNumber == ""):
            return f"Average attempts per patient"
        Bs = self._fetch("B", paramRowNumber, par1, par2)
        result = []
        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if Bs[i].empty:
                        result.append([None,None])
                    else:
                        result.append(np.quantile(Bs[i].sum(1), [0.25, 0.75]))
                if self.var == "sd":
                    result.append([np.std(Bs[i].sum(1)), None])
        else:
            for i in range(0, 3):
                result.append(Bs[i].sum(1).mean())
        return (result)

    def ratio_succes_unsuccess(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ## OUTPUT:
        ## The ratio of successful : not succsesful attempts
        ## If inputs are empty, return function descriptor
        if (paramRowNumber == ""):
            return "Ratio succesful to unsuccessful attempts"
        Bs = self._fetch("B", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            success = Cs[i].sum(1).sum()
            unsuccess = Bs[i].sum(1).sum() - success
            result.append(success / unsuccess)
        return (result)

    def avg_visits(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ## OUTPUT:
        ## The average number of visits per patient
        if (paramRowNumber == ""):
            return "Average visits per patient"
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []

        if no_group:
            alltogether = np.concatenate((Cs[0], Cs[1],Cs[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        result.append(np.quantile(alltogether.sum(1)), [0.25, 0.75])
                    else:
                        result.append(np.quantile(Cs[i].sum(1)), [0.25, 0.75])
                if self.var == "sd":
                    if no_group:
                        result.append(np.std(alltogether.sum(1)), None)
                    else:
                        result.append(np.std(Cs[i].sum(1)), None)
        else:
            for i in range(0, 3):
                if no_group:
                    result.append(alltogether.sum(1).mean())
                else:
                    result.append(Cs[i].sum(1).mean())
        return (result)

    def prop_success(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ## OUTPUT:
        ## The proportion of successful attempts
        ## If inputs are empty, return function descriptor
        if (paramRowNumber == ""):
            return "Ratio of succesful to total attempts"
        Bs = self._fetch("B", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            success = Cs[i].sum(1).sum()
            attempts = Bs[i].sum(1).sum()
            result.append(success / attempts)
        return (result)

    def unsuccessful_attempts_patient(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ## OUTPUT:
        ## Average number of unsuccesful attempts among all patients who attempted to get a visit
        ## If inputs are empty, return function descriptor
        if (paramRowNumber == ""):
            return "Unsuccessful attempts per patient"
        Bs = self._fetch("B", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            result.append(((Bs[i].sum(1) - Cs[i].sum(1))).mean())
        return (result)

    ####
    #### Performance metrics
    ####

    def _satisficed_needs(self, N):
        ##OUTPUT
        ## An array of needs satisficed by patient
        diferences = np.diff(N, axis=1)
        return (diferences * (diferences < 0)).sum(1) * -1

    def _generated_needs(self, N):
        ##OUTPUT
        ## An array of needs generated by patient including needs in the first week
        diferences = np.diff(N * -1, axis=1)
        total_generated = (diferences * (diferences < 0)).sum(1) * -1
        initial_needs = N.iloc[:, 0]  ## present at N1, not summed above
        return total_generated + initial_needs

    def _visits(self, C):
        ##OUTPUT
        ## Returns an array of visits per patient
        return C.sum(1)

    def _not_satisficed_needs(self, N):
        ##The same as the last colum of the Ns, BUT I'm not sure about the noise in disease progress (should cancell the same, I haven't tested it)
        ##OUTPUT
        ## An array of needs that are not satisficed by patient
        return (self._generated_needs(N) - self._satisficed_needs(N))

    def not_satisfied_needs_patient_simulation(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ##OUTPUT
        ## The average not-satisfied needs at the end of the simulation
        if (paramRowNumber == ""):
            return "Unsatisfied needs per patient at the end"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            result.append(self._not_satisficed_needs(Ns[i]).mean())
        return result

    def ratio_satisfied_visits(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ##OUTPUT
        ## The average satisfied needs per visit
        if (paramRowNumber == ""):
            return "Ratio satisfieds to visits"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            total_satisfied = self._satisficed_needs(Ns[i]).sum()
            total_visits = self._visits(Cs[i]).sum()
            result.append(total_satisfied / total_visits)
        return result

    def ratio_visits_satisfied(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ##OUTPUT
        ## The average satisfied needs per visit
        if (paramRowNumber == ""):
            return "Ratio visits to satisfieds"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            total_satisfied = self._satisficed_needs(Ns[i]).sum()
            total_visits = self._visits(Cs[i]).sum()
            result.append(total_visits / total_satisfied)
        return result

    def ratio_not_satisfied_visits(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ##OUTPUT
        ## The average satisfied needs per visit
        if (paramRowNumber == ""):
            return "Ratio not-satisfieds to visits"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            total_not_satisfied = self._not_satisficed_needs(Ns[i]).sum()
            total_visits = self._visits(Cs[i]).sum()
            result.append(total_not_satisfied / total_visits)
        return result

    def satisfaction_perVisit_perPatient(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        ##OUTPUT
        ## An averege among patients of the needs solved per visit
        if (paramRowNumber == ""):
            return "Needs satisfied per visit per patient"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            vis_patient = self._visits(Cs[i])
            if (vis_patient.sum() == 0):
                result.append(0)
            else:
                sat_patient = self._satisficed_needs(Ns[i])
                result.append((sat_patient[vis_patient != 0] / vis_patient[vis_patient != 0]).mean())
        return (result)

    def need_evolution(self, paramRowNumber="", par1="", par2="",  variability=False, no_group=False):
        ##OUTPUT
        ## The average difference between H1 and H(timeHorizon) among patients
        if (paramRowNumber == ""):
            return "Average needs change from begining to end"
        Ns = self._fetch("H", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Ns[0], Ns[1],Ns[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        initial = alltogether[:,0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.quantile(end - initial), [0.25, 0.75])
                    else:
                        initial = Ns[i].iloc[:,0]
                        end = Ns[i].iloc[:, Ns[i].shape[1] - 1]
                        result.append(np.quantile(end - initial), [0.25, 0.75])

                if self.var == "sd":
                    if no_group:
                        initial = alltogether[:,0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.std(end - initial), None)
                    else:
                        initial = Ns[i].iloc[:,0]
                        end = Ns[i].iloc[:, Ns[i].shape[1] - 1]
                        result.append(np.std(end - initial), None)
        else:
            for i in range(0, 3):
                if no_group:
                    initial = alltogether[:,0]
                    end = alltogether[:, alltogether.shape[1] - 1]
                    result.append((end - initial).mean())
                else:
                    initial = Ns[i].iloc[:,0]
                    end = Ns[i].iloc[:, Ns[i].shape[1] - 1]
                    result.append((end - initial).mean())
        return result

    def need_evolution_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        result = self.need_evolution(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = True)
        return result

    def expectation_evolution(self, paramRowNumber="",par1="", par2="",  variability=False, no_group=False):
        ##OUTPUT
        ## The average difference between H1 and H(timeHorizon) among patients
        if (paramRowNumber == ""):
            return "Average expectation change from begining to end"
        Es = self._fetch("exp", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Es[0], Es[1], Es[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        initial = alltogether[:, 0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.quantile(end - initial), [0.25, 0.75])
                    else:
                        initial = Es[i].iloc[:, 0]
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.quantile(end - initial), [0.25, 0.75])

                if self.var == "sd":
                    if no_group:
                        initial = alltogether[:, 0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.std(end - initial), None)
                    else:
                        initial = Es[i].iloc[:, 0]
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.std(end - initial), None)
        else:
            for i in range(0, 3):
                if no_group:
                    initial = alltogether[:, 0]
                    end = alltogether[:, alltogether.shape[1] - 1]
                    result.append((end - initial).mean())
                else:
                    initial = Es[i].iloc[:, 0]
                    end = Es[i].iloc[:, Es[i].shape[1] - 1]
                    result.append((end - initial).mean())
        return result

    def expectation_evolution_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        result = self.expectation_evolution(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = True)
        return result

    def capacity_usage(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        if (paramRowNumber == ""):
            return "Percentaje of capacity used "
        totalCap = self.paramGrid.loc[paramRowNumber]["capacity"] * self.paramGrid.loc[paramRowNumber]["weeks"]
        if totalCap == 0:
            return [0, 0, 0]

        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []

        for i in range(0, 3):
            result.append(Cs[i].sum().sum() / totalCap)
        return (result)


    ####
    #### Others
    ####

    def mean_expectations(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        if (paramRowNumber == ""):
            return "Average expectations per patient"
        Es = self._fetch("exp", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Es[0], Es[1],Es[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        result.append(np.quantile(alltogether.mean(1), [0.25, 0.75]))
                    else:
                        result.append(np.quantile(Es[i].mean(1), [0.25, 0.75]))
                if self.var == "sd":
                    if no_group:
                        result.append([np.std(alltogether.mean(1)), None])
                    else:
                        result.append([np.std(Es[i].mean(1)), None])
        else:
            for i in range(0, 3):
                if no_group:
                    result.append(alltogether.mean(1).mean())
                else:
                    result.append(Es[i].mean(1).mean())
        return result

    def mean_expectations_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=True):
        result = self.mean_expectations(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = no_group)
        return result


    def mean_treatment(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        if (paramRowNumber == ""):
            return "Average treatment per patient during the simulation"
        Ts = self._fetch("T", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Ts[0], Ts[1],Ts[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        result.append(np.quantile(alltogether.mean(1), [0.25, 0.75]))
                    else:
                        result.append(np.quantile(Ts[i].mean(1), [0.25, 0.75]))
                if self.var == "sd":
                    if no_group:
                        result.append([np.std(alltogether.mean(1)), None])
                    else:
                        result.append([np.std(Ts[i].mean(1)), None])
        else:
            for i in range(0, 3):
                if no_group:
                    result.append(alltogether.mean(1).mean())
                else:
                    result.append(Ts[i].mean(1).mean())
        return result


    def mean_treatment_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=True):
        result = self.mean_treatment(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = no_group)
        return result


    def mean_needs(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        if (paramRowNumber == ""):
            return "Average needs per patient"
        Hs = self._fetch("H", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Hs[0], Hs[1],Hs[2]))
        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        result.append(np.quantile(alltogether.mean(1), [0.25, 0.75]))
                    else:
                        result.append(np.quantile(Hs[i].mean(1), [0.25, 0.75]))
                if self.var == "sd":
                    if no_group:
                        result.append([np.std(alltogether.mean(1)), None])
                    else:
                        result.append([np.std(Hs[i].mean(1)), None])
        else:
            for i in range(0, 3):
                if no_group:
                    result.append(alltogether.mean(1).mean())
                else:
                    result.append(Hs[i].mean(1).mean())
        return (result)

    def mean_needs_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        result = self.mean_needs(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = True)
        return result

    def visits_per_patient(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        if (paramRowNumber == ""):
            return "Number of visits per patient"
        Cs = self._fetch("C", paramRowNumber, par1, par2)
        result = []
        for i in range(0, 3):
            result.append(self._visits(Cs[i]))
        return (result)

    def expectation_end_sim(self, paramRowNumber="", par1="", par2="",  variability=False, no_group=False):
        """Compute the average and variability of last week's expectations at the end of the simulation"""
        if (paramRowNumber == ""):
            return "Average expectations at the end of the simulation"
        Es = self._fetch("exp", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Es[0], Es[1],Es[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.quantile(end), [0.25, 0.75])
                    else:
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.quantile(end), [0.25, 0.75])

                if self.var == "sd":
                    if no_group:
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.std(end), None)
                    else:
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.std(end), None)
        else:
            for i in range(0, 3):
                if no_group:
                    end = alltogether[:, alltogether.shape[1] - 1]
                    result.append((end).mean())
                else:
                    end = Es[i].iloc[:, Es[i].shape[1] - 1]
                    result.append((end).mean())
        return result

    def mean_progress(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        if (paramRowNumber == ""):
            return "Average needs generated per patient (disease)"
        Ps = self._fetch("P", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Ps[0], Ps[1],Ps[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        result.append(np.quantile(alltogether.mean(1), [0.25, 0.75]))
                    else:
                        result.append(np.quantile(Ps[i].mean(1), [0.25, 0.75]))
                if self.var == "sd":
                    if no_group:
                        result.append([np.std(alltogether.mean(1)), None])
                    else:
                        result.append([np.std(Ps[i].mean(1)), None])
        else:
            for i in range(0, 3):
                if no_group:
                    result.append(alltogether.mean(1).mean())
                else:
                    result.append(Ps[i].mean(1).mean())
        return (result)

    def last_step_needs(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        """Return the needs for all patients during the last iteration"""
        Hs = self._fetch("H", paramRowNumber, par1, par2)
        n_steps = Hs[0].shape[1]-1
        return Hs[0].loc[:,n_steps], Hs[1].loc[:,n_steps], Hs[2].loc[:,n_steps]

    def last_step_exp(self, paramRowNumber="", par1="", par2="", variability=False, no_group=False):
        """Return the expectations for all patients during the last iteration"""
        Es = self._fetch("exp", paramRowNumber, par1, par2)
        n_steps = Es[0].shape[1]-1
        return Es[0].loc[:,n_steps], Es[1].loc[:,n_steps], Es[2].loc[:,n_steps]

    def expectation_endstage(self, paramRowNumber="",par1="", par2="",  variability=False, no_group=False):
        ##OUTPUT
        ## The average difference between H1 and H(timeHorizon) among patients
        if (paramRowNumber == ""):
            return "Average expectation change from begining to end"
        Es = self._fetch("exp", paramRowNumber, par1, par2)
        result = []
        if no_group:
            alltogether = np.concatenate((Es[0], Es[1], Es[2]))

        if variability:
            for i in range(0, 3):
                if self.var == "quantile":
                    if no_group:
                        #initial = alltogether[:, 0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.quantile(end), [0.25, 0.75])
                    else:
                        #initial = Es[i].iloc[:, 0]
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.quantile(end), [0.25, 0.75])

                if self.var == "sd":
                    if no_group:
                        #initial = alltogether[:, 0]
                        end = alltogether[:, alltogether.shape[1] - 1]
                        result.append(np.std(end), None)
                    else:
                        #initial = Es[i].iloc[:, 0]
                        end = Es[i].iloc[:, Es[i].shape[1] - 1]
                        result.append(np.std(end), None)
        else:
            for i in range(0, 3):
                if no_group:
                    #initial = alltogether[:, 0]
                    end = alltogether[:, alltogether.shape[1] - 1]
                    result.append((end).mean())
                else:
                    #initial = Es[i].iloc[:, 0]
                    end = Es[i].iloc[:, Es[i].shape[1] - 1]
                    result.append((end).mean())
        return result

    def expectation_endstage_nogroup(self, paramRowNumber="", par1="", par2="", variability=False,no_group=False):
        result = self.expectation_endstage(paramRowNumber=paramRowNumber, par1=par1, par2=par2, variability=variability, no_group = True)
        return result
