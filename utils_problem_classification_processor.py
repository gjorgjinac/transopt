from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sklearn 
from utils_pre_processor import PreProcessor
import seaborn as sns
import pandas as pd
import numpy as np
import random

class ProblemClassificationProcessor():
    verbose:bool
    normalize:bool
    reduce:bool
    scaler:any
    dimension_reducer:any
    fold:int
    
    def __init__(self, verbose=False, normalize=True, reduce=False, fold=None, split_ids_dir=None):
        self.pre_processor = PreProcessor(verbose=verbose, normalize=normalize, reduce=reduce)
        self.verbose=verbose
        self.fold=fold
    
    def get_x_y(self, sample_df, shuffle=True):
        if self.verbose:
            print('Extracting x and y')
        xys=[]
        problem_ids = sample_df.index.get_level_values('problem_id').drop_duplicates().sort_values().values
        instance_ids = sample_df.index.get_level_values('instance_id').drop_duplicates().sort_values().values
        for problem_id in problem_ids:
            problem_samples=sample_df.query("problem_id==@problem_id")
            for instance_id in instance_ids:
                problem_instance_samples=problem_samples.query("instance_id==@instance_id")
                xys+=[(problem_instance_samples.values, problem_id, (problem_id, instance_id))]
        if shuffle:
            random.shuffle(xys)
        x=np.array([xy[0] for xy in xys])
        y=np.array([xy[1] for xy in xys])
        ids=np.array([xy[2] for xy in xys])
        return x,y,ids

    
    def split_data(self, sample_df):
        if self.fold==None:
            if self.verbose:
                print('Splitting data')
            instance_ids = list(set(sample_df.index.get_level_values('instance_id').values))
            train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.2)
            train_instance_ids, val_instance_ids = train_test_split(train_instance_ids, test_size=0.2)
        else:
            if self.verbose:
                print('Reading split data')
            max_instance_id=sample_df.index.get_level_values('instance_id').max()
            train_instance_ids, val_instance_ids, test_instance_ids = [list(pd.read_csv(f'folds/problem_classification_{max_instance_id}_instances/{split_name}_{self.fold}.csv',index_col=[0])['0'].values) for split_name in ['train','val','test']]
        train, val, test =[sample_df.query(f"instance_id in @split_instance_ids").sample(frac=1) for split_instance_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]

        return train, val, test

    def offset_problem_id(self, sample_df):
        problem_ids = sample_df.index.get_level_values('problem_id').values
        unique_problem_ids=np.unique(problem_ids)
        problem_id_to_index={problem_id: problem_id_index for problem_id_index, problem_id in enumerate(list(unique_problem_ids))}
        problem_index_to_id={problem_id_index: problem_id for problem_id, problem_id_index in problem_id_to_index.items()}

        new_problem_ids = [problem_id_to_index[p] for p in problem_ids]
        index_names=list(sample_df.index.names)
        sample_df = sample_df.reset_index()
        sample_df['problem_id']=new_problem_ids
        sample_df = sample_df.set_index(index_names)
        return sample_df, problem_id_to_index,  problem_index_to_id

    
    
    def run(self,sample_df):
        sample_df, problem_id_to_index,  problem_index_to_id = self.offset_problem_id(sample_df)
        train, val, test = self.split_data(sample_df)
        train, val, test = self.pre_processor.preprocess_splits(train, val, test)
        x_train, y_train, ids_train= self.get_x_y(train)
        x_val,y_val, ids_val=self.get_x_y(val)
        x_test, y_test, ids_test = self.get_x_y(test)
        return x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test
