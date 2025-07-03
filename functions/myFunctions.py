import numpy as np
import pandas as pd

def gridCombined(params):
    # returns the final grid of params as a dataframe taking care of preserving data types
    values = np.array(combinations(params)).T
    return (pd.DataFrame(data=values, columns=par_key(params)).astype(type_args(params)))


def combinations(dic):
    # generates ALL the possible combinations of the given parameters
    arr = par_val(dic)
    seq_long = num_reps(arr)  ##number of parameters combinations
    int_copy = [x for x in arr]
    result = []
    for i in arr:
        current, rest = int_copy.pop(0), int_copy
        new_line = gen_row(current, num_reps(rest))
        new_line = new_line * int(seq_long / len(new_line))  ##adjust the size of the new line
        result.append(new_line)
    return result


def gen_row(current, n_reps):
    # repete a the sequence, n_reps are the number of consecutive characters
    seq = []
    for i in range(len(current)):
        for ii in range(n_reps):
            seq.append(current[i])
    return seq


def num_reps(arr):
    # how many differents values will be combined down the array
    t = 1
    for i in arr:
        t *= len(i)
    return (t)


def par_val(dic):
    ##extract the values to a list
    result = []
    for i in dic.values():
        result.append(i)
    return (result)


def par_key(dic):
    ##extract the keys to a list
    result = []
    for i in dic.keys():
        result.append(i)
    return (result)


def type_args(params):
    # Creates the arguments type for the Panda DatFrame, to preserve arguments types (they get converted to float, apparently by Numpy in the back)
    i = 1
    types = {}
    for key, value in params.items():
        types[key] = ret_type(value[0])
        i += 1
    return (types)


def ret_type(value):
    # returns a string with the data type
    if type(value) == float:
        return "float"
    if type(value) == int:
        return "int"
    #else:
    #    print("unknown parameter data type")






