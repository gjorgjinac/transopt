import utils_runner_universal
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys


arguments=sys.argv

dimension = int(arguments[1])
sample_count_dimension_factor=int(arguments[2])
instances_to_use=int(arguments[3])


x_columns=[f'x_{i}' for i in range (0,dimension)]
y_columns = ['y']



def read_sample_df(dimension, sample_count_dimension_factor):
    file_to_read = open(f"data/lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.p", "rb")
    loaded_dictionary = pickle.load(file_to_read)
    all_sample_df=pd.DataFrame.from_dict(loaded_dictionary,orient='index')
    all_sample_df.columns=x_columns+y_columns
    all_sample_df.index=pd.MultiIndex.from_tuples(all_sample_df.index, names=['problem_id', 'instance_id','dimension','sample_id'])
    return all_sample_df

def scale_y(all_sample_df):
    new_sample_df=pd.DataFrame()
    for problem_id in all_sample_df.reset_index()['problem_id'].drop_duplicates():
        problem_instances=all_sample_df.loc[problem_id]
        for instance_id in all_sample_df.reset_index()['instance_id'].drop_duplicates():

            instance_df=problem_instances.loc[instance_id].copy()
            min_max_scaler = MinMaxScaler()

            y_scaled = min_max_scaler.fit_transform(instance_df['y'].values.reshape(-1, 1))
            instance_df.loc[:,'y']=y_scaled
            instance_df['problem_id']=problem_id
            instance_df['instance_id']=instance_id
            new_sample_df=pd.concat([new_sample_df,instance_df])
            
    new_sample_df=new_sample_df.reset_index(drop=True).set_index(['problem_id','instance_id'])
    return new_sample_df

scaled_file=f"data/scaled_lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.csv"
if not os.path.isfile(scaled_file):

    all_sample_df=read_sample_df(dimension, sample_count_dimension_factor)
    all_sample_df=all_sample_df.query("instance_id<=@instances_to_use")
    new_sample_df=scale_y(all_sample_df)
    new_sample_df.to_csv(scaled_file)
else:
    new_sample_df=pd.read_csv(scaled_file, index_col=[0,1])

for fold in range(0,10):
    for d_model in [30,50,100]:
        for n_heads in [1,3]:
            for n_layers in [1,3]:
                    sample_df=new_sample_df.copy()
                    utils_runner_universal.UniversalRunner(extra_info=f'dim_{dimension}_instances_{instances_to_use}_samples_{sample_count_dimension_factor}', fold=fold, verbose=True, plot_training=False, d_model=d_model, d_k=None, d_v=None, n_heads=n_heads, n_layers=n_layers, n_epochs=200,  lr_max=0.001).run(sample_df)






