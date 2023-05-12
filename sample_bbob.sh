#!/bin/bash

for sample_count_factor in 50 100
do

    for dimension in 3 5 10 20
    do
        tsp python sample_bbob_lhs.py $dimension $sample_count_factor
    done

done