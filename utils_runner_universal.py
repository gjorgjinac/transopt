from tsai_custom import *
from tsai.all import *
computer_setup()
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import sklearn 
from fastai.callback.tracker import EarlyStoppingCallback
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from utils_problem_classification_processor import *


from model_stats import *
from sklearn.metrics.pairwise import cosine_similarity
import os
from config import *

import colorcet as cc
import matplotlib.pyplot as plt


import matplotlib
matplotlib.cm.register_cmap("my_cmap", my_cmap)

class UniversalRunner():
    logger = None
    verbose: bool
    plot_training: bool
    n_heads:int 
    n_layers:int 
    d_model:int
    d_k:int
    d_v:int 
    n_epochs:int
    bs:int

    model:any
    learner:any

    result_dir:str

    
    def __init__(self,task_name='problem_classification',extra_info=None, verbose=True, lr_max=5e-4, plot_training=True,  n_heads=1, n_layers=1, d_model=20, d_k=10, d_v=10, n_epochs=100, batch_size=8, fold=None, split_ids_dir=None, global_result_dir='results', aggregations=None):
        """
        Initialize the UniversalRunner object with the given parameters and create the data processor object.

        Parameters
        task_name : str
        The name of the task to run the model on, such as 'problem_classification' or 'algorithm_classification'.
        extra_info : str
        An optional string to add extra information to the setting name and result directory.
        verbose : bool
        A flag to indicate whether to print verbose messages or not.
        lr_max : float
        The maximum learning rate to use for training the model.
        plot_training : bool
        A flag to indicate whether to plot the training metrics or not.
        n_heads : int
        The number of attention heads in the transformer model.
        n_layers : int
        The number of encoder layers in the transformer model.
        d_model : int
        The dimension of the input and output vectors in the transformer model.
        d_k : int
        The dimension of the query and key vectors in the attention mechanism.
        d_v : int
        The dimension of the value vector in the attention mechanism.
        n_epochs : int
        The number of epochs to train the model.
        batch_size : int
        The batch size for the data loaders.
        fold : int or None
        The fold number to use for cross-validation or None for no cross-validation.
        split_ids_dir : str or None
        The path to the directory where the split ids are stored or None for random splits.
        global_result_dir : str
        The path to the directory where all results are stored.
        aggregations : list of str or None
        The list of aggregation functions to use for aggregating the transformer embeddings. Can contain a list of the following aggregations "min", "max", "mean" or "std"
        """
        self.verbose = verbose
        self.lr_max=lr_max
        self.plot_training=plot_training
        self.n_heads=n_heads
        self.n_layers=n_layers
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.n_epochs=n_epochs
        self.batch_size=batch_size
  

    
        self.task_name=task_name
        self.extra_info=(extra_info if extra_info is not None else '') + f'_fold_{fold if fold is not None else "none"}_n_heads_{n_heads}_n_layers_{n_layers}_d_model_{d_model}_d_k_{d_k}_d_v_{d_v}_aggregations_{"all" if aggregations is None else "-".join(aggregations)}'

        self.result_dir = os.path.join(global_result_dir,self.task_name,self.extra_info) if fold is None or task_name=='problem_classification' else os.path.join(global_result_dir,self.task_name,self.extra_info, split_ids_dir)
        self.fold=fold

        self.aggregations=aggregations
        os.makedirs(self.result_dir, exist_ok = True) 
        
        self.data_processor= ProblemClassificationProcessor(verbose=False, normalize=False, reduce=False,fold=fold,split_ids_dir=split_ids_dir)  

        
        
        self.classification_report_location=os.path.join(self.result_dir,f'test_classification_report.csv')
      


    def find_learning_rate(self, dls):
        print('Determining learning rate')
        learn = Learner(dls, self.model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAuc(), accuracy],  cbs=ShowGraphCallback2())
        learn.lr_find()


    def train_model(self, dls):
        print('Training model')
        callbacks=[EarlyStoppingCallback( min_delta=0.001,patience=5)]
        #callbacks=[]
        if self.plot_training:
            callbacks+=[ShowGraphCallback2()]

        self.learner = Learner(dls, self.model, loss_func=CrossEntropyLoss(), metrics=[ accuracy],  cbs=callbacks)
        start = time.time()
        self.learner.fit(self.n_epochs, lr=self.lr_max)
 
        print('\nElapsed time:', time.time() - start)
        if self.plot_training:
            self.learner.plot_metrics()


    def evaluate(self, x, y, split_name,  problem_index_to_id, ids ):
        print('Evaluating model')
        probas, targets, preds = self.learner.get_X_preds(np.swapaxes(x,1,2), with_decoded=True)
        predicted_classes = [int(p.argmax()) for p in probas]
        if problem_index_to_id is not None:
            y = [problem_index_to_id[yy] for yy in y]
            predicted_classes = [problem_index_to_id[yy] for yy in predicted_classes]
            
        
        report_df=pd.DataFrame(classification_report(y,predicted_classes, output_dict=True))
        report_df.to_csv(self.classification_report_location)
        print(report_df)
        confusion_matrix_df=pd.DataFrame(confusion_matrix(y, predicted_classes))
        confusion_matrix_df.to_csv(os.path.join(self.result_dir,f'{split_name}_confusion_matrix.csv'))
        
        ConfusionMatrixDisplay.from_predictions(y, predicted_classes, cmap=my_cmap)
        
        
        
        d=pd.DataFrame([y, predicted_classes]).T
        print(ids)
        id_names=["problem_id","instance_id"] if self.task_name=='problem_classification'  else ["problem_id","instance_id", "seed", "algorithm_name"] 
        d.index= pd.MultiIndex.from_tuples([tuple(i) for i in ids], names=id_names )
        #d.index=ids
        d.columns=['ys','predictions']
        d.to_csv(os.path.join(self.result_dir,f'{split_name}_ys_predictions.csv'))

        plt.savefig(os.path.join(self.result_dir,f'{split_name}_confusion_matrix.pdf'))
        return report_df, confusion_matrix_df

    def get_batch_embeddings(self, batch):
        batch_embeddings=self.model.cuda().get_embeddings(batch[0].cuda())
        batch_embeddings=batch_embeddings.detach().cpu().numpy()
        
        #np.swapaxes(batch_embeddings,1,2)
        return batch_embeddings


    def get_embeddings_from_dls(self, dls, batch_count=None, cast_y_to_int=True):
        #batch = dls.one_batch()
        all_embeddings=None
        all_labels=[]
        i=0
        for batch in dls:
            batch_embeddings=self.get_batch_embeddings(batch)
            all_embeddings=batch_embeddings if all_embeddings is None else np.append(all_embeddings,batch_embeddings, axis=0)
            all_labels+=list(batch[1])
            i+=1
            if batch_count is not None and i >= batch_count:
                break
        if cast_y_to_int:
            all_labels=[int(i) for i in all_labels]
        return all_embeddings, all_labels

    
            
    def shuffle_x_y(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def preprocess(self,sample_df):
        x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test = self.data_processor.run(sample_df)
        
        x_train, y_train = self.shuffle_x_y(x_train, y_train)
        x_val,y_val = self.shuffle_x_y(x_val,y_val)
        
        return x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id
    
    def save_embeddings(self, sample_df):
        x,y,ids=self.data_processor.get_x_y(sample_df, shuffle=False)
        dset = TSDatasets(np.swapaxes(x,1,2),sample_df.index.drop_duplicates())
        dls = TSDataLoaders.from_dsets(dset, bs=24, shuffle=False)
        embeddings=self.get_embeddings_from_dls( dls[0], None, cast_y_to_int=False)


        embedding_df=pd.DataFrame(embeddings[0])
        embedding_df[['problem_id','instance_id']]=[tuple([int(cc) for cc in c.cpu()]) for c in embeddings[1]]
        embedding_df=embedding_df.set_index(['problem_id','instance_id'])
        embedding_df.to_csv(os.path.join(self.result_dir,f'embeddings.csv'), compression='zip')
        return embedding_df

    def run (self,sample_df, plot_embeddings=False, save_embeddings=True, regenerate=False):
        if os.path.isfile(self.classification_report_location) and not regenerate:
            print(f'Classification report already exists {self.classification_report_location}. Skipping execution.')
            return None, None, None, None, None
            
        x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test = self.data_processor.run(sample_df)
        dset_train, dset_val, dset_test = [TSDatasets(np.swapaxes(xx,1,2),yy) for xx,yy in [(x_train,y_train),(x_val,y_val),(x_test,y_test)]]
        
        dls = TSDataLoaders.from_dsets(dset_train, dset_val, bs=self.batch_size)
        
        test_data_loader = TSDataLoaders.from_dsets(dset_test, bs=self.batch_size)[0]
        
        dls.c=len(set(y_train))
        print(f'Number of samples: {dls.len}, Number of variables: {dls.vars}, Number of classes: {dls.c}')
        self.model=OptTransStats(dls.vars, dls.c, dls.len, n_heads=self.n_heads, n_layers=self.n_layers, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, use_positional_encoding=False, aggregations=self.aggregations)

        self.train_model(dls)
        self.evaluate(x_train,y_train,'train', problem_index_to_id, ids_train )
        self.evaluate(x_val,y_val,'val', problem_index_to_id, ids_val)
        test_report, test_confusion_matrix = self.evaluate(x_test,y_test,'test', problem_index_to_id, ids_test )
  
        embedding_df=None
        if save_embeddings:
            embedding_df=self.save_embeddings(sample_df)
        #self.check_random_forest_performance(dset_train, dset_val, dset_test)
        
        torch.save(self.model, os.path.join(self.result_dir,f'trained_model.pt'))
        return self.model, self.learner, test_report, test_confusion_matrix, embedding_df