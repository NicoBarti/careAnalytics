from scripts.pairPlots import *
import matplotlib.pyplot as plt
from MyClasses.plotting import *
from scripts.needs_to_attemps import *
import numpy as np
from scipy.stats import alpha

from scripts.pairPlots import *
from MyClasses.client import *
from scripts.needs_to_attemps import *

def do_expectations_converge(c):
    par = {"capacity": [80], "DISEASE_SEVERITY": [3], "LEARNING_RATE": [0],
           "SUBJECTIVE_INITIATIVE": [1],
           "numPatients": [1000], "weeks": [1500]}
    parGrid = gridCombined(par)
    c.start_server()
    error = Errors_socket(gridParameters = parGrid, N = 10, client = c,
                          reuse_previous_output=False, vervoso=False)

    print('doing expectations_converge')
