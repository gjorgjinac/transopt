#!/bin/bash

for sample_count_factor in 50 100
do

    for dimension in 3 5 10 20
    do
        python sample_bbob_lhs.py $dimension $sample_count_factor
        
        python static_problem_classification_parameters_exp.py $dimension $sample_count_factor 999
        python static_problem_classification_parameters_downstream.py $dimension $sample_count_factor 999
        python static_problem_classification_parameters_aggregations.py $dimension $sample_count_factor 999
    done

done


