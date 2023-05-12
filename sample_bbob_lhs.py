import pickle

import numpy as np
from cocoex import Suite
from smt.sampling_methods import LHS
import sys

arguments=sys.argv
suite = Suite('bbob', "instances: 1-999", '')

dataframe = {}
dimension = int(arguments[1])
sample_count_dimension_factor=int(arguments[2])
xlimits = np.array([[-5,5] for _ in range(0,dimension)])
sampling = LHS(xlimits=xlimits)


sample_count = sample_count_dimension_factor * dimension
samples = sampling(sample_count)

for f_id in range(1, 25):
    print(f_id)

    for instance in range(1,1000):
        f = suite.get_problem_by_function_dimension_instance(f_id, dimension, instance)
        print(f.info)
        for x_index, x in enumerate(samples):
            x_y = list(x).copy()
            x_y.append(f(x))
            dataframe[(f.id_function, f.id_instance, dimension, x_index)] = x_y



pickle.dump( dataframe, open( f"data/lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.p", "wb" ) )