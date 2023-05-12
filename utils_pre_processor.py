from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sklearn 

import seaborn as sns
import pandas as pd
import numpy as np
import random

class PreProcessor():
    verbose:bool
    normalize:bool
    reduce:bool
    scaler:any
    dimension_reducer:any
    
    def __init__(self, verbose=False, normalize=True, reduce=False):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.dimension_reducer = PCA(1)
        self.normalize=normalize
        self.reduce=reduce
        self.verbose=verbose

    
    
    def normalize_y(self, df):
        print('Normalizing')
        print(df.shape)
        new_sample_df=pd.DataFrame()
        df=df.reset_index()
        run_ids=df[['algorithm_name','problem_id','instance_id','seed']].drop_duplicates().values
        df=df.set_index(['algorithm_name','problem_id','instance_id','seed','iteration'])
        df=df.sort_index()
        #x_columns = [f'x_{i}' for i in range(0,df.shape[1]-1)]
        #df.columns=x_columns + ['y']
        for algorithm_name,problem_id,instance_id,seed in run_ids:
            min_max_scaler = MinMaxScaler()
            trajectory_scaled=df.loc[(algorithm_name,problem_id,instance_id,seed,)]
            y_scaled = min_max_scaler.fit_transform(trajectory_scaled['y'].values.reshape(-1, 1))
            trajectory_scaled['y']=y_scaled
            trajectory_scaled[['algorithm_name','problem_id','instance_id','seed']]=algorithm_name,problem_id,instance_id,seed

            new_sample_df=pd.concat([new_sample_df,trajectory_scaled.reset_index(drop=False)])

        new_sample_df=new_sample_df.set_index(['algorithm_name','problem_id','instance_id','seed','iteration'])
        

        return new_sample_df
    
    def split_preprocessing(self, split,split_name):
        if self.verbose:
            print('Doing preprocessing')
        x_columns = [f'x_{i}' for i in range(0,split.shape[1]-1)]
        y_columns=['y']
        split_og_index=split.copy().index
        split_og_columns = split.copy().columns
        split.columns=x_columns+y_columns
        #if split_name=='train':
        #self.scaler.fit(split['y'].values.reshape(-1, 1))
        #self.dimension_reducer.fit(split[x_columns])
        if self.normalize:
            split=self.normalize_y(split)
            #split['y'] = self.scaler.transform(split['y'].values.reshape(-1, 1))
            #split = pd.DataFrame(split, index=split_og_index, columns=split_og_columns)
        if self.reduce:
            split_x = self.dimension_reducer.transform(split[x_columns])
            split=np.concatenate((split_x, split[y_columns]),axis=1)
        #split = pd.DataFrame(split, index=split_og_index)
        return split


    def preprocess_splits(self, train, val, test):
        if self.normalize or self.reduce:
            train = self.split_preprocessing(train,'train')
            val = self.split_preprocessing(val,'val')
            test = self.split_preprocessing(test,'test')
        return train, val, test
