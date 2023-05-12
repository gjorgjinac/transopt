#!/bin/bash

for sample_count_factor in 50 100 200
do

    for dimension in 3 5 10 20
    do
        for instance_count in 100 999
        do
            tsp python static_problem_classification_parameters_exp.py $dimension $sample_count_factor $instance_count
        done
    done

done