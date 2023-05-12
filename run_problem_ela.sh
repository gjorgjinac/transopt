#!/bin/bash

for problem_id in {1..24}
do

    tsp Rscript feature_calculation_per_problem.R lhs_samples_dimension_20_50_samples.csv $problem_id

done