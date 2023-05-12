from tsai_gina import *
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
    model_name:str
    model:any
    learner:any
    setting_name:str
    result_dir:str
    use_positional_encoding:bool
    
    def __init__(self,model_name,task_name='problem_classification',extra_info=None, y_mapping=None,normalize=True,reduce=False, verbose=True, lr_max=5e-4, plot_training=True, use_positional_encoding=False, n_heads=1, n_layers=1, d_model=20, d_k=10, d_v=10, n_epochs=100, batch_size=8, fold=None, split_ids_dir=None, iteration_count=None, include_iteration_in_x=False, sample_range=None, iteration_based=True, train_seeds=None, val_seeds=None, global_result_dir='results', aggregations=None):

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
        self.train_seeds=train_seeds
        self.val_seeds=val_seeds
        self.setting_name=f'{task_name}_{model_name}'
        self.task_name=task_name
        self.extra_info=(extra_info if extra_info is not None else '') + f'_fold_{fold if fold is not None else "none"}_n_heads_{n_heads}_n_layers_{n_layers}_d_model_{d_model}_d_k_{d_k}_d_v_{d_v}_aggregations_{"all" if aggregations is None else "-".join(aggregations)}'
        if task_name=='algorithm_classification':
            self.extra_info+=f'_pos_enc_{use_positional_encoding}'
        self.result_dir = os.path.join(global_result_dir,self.setting_name,self.extra_info) if fold is None or task_name=='problem_classification' else os.path.join(global_result_dir,self.setting_name,self.extra_info, split_ids_dir)
        self.fold=fold
        self.use_positional_encoding=use_positional_encoding
        self.iteration_count=iteration_count
        self.aggregations=aggregations
        os.makedirs(self.result_dir, exist_ok = True) 
        
        self.data_processor= ProblemClassificationProcessor(verbose=False, normalize=normalize, reduce=reduce,fold=fold,split_ids_dir=split_ids_dir)  
        self.model_name=model_name
        
        
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

    def show_cosine_similarity(self,embeddings,labels, problem_index_to_id, count=10):
        plt.figure(figsize=(10,10)) 
        if problem_index_to_id is not None:
            labels = [problem_index_to_id[yy] for yy in labels]
        embeddings, labels = embeddings[:count], labels[:count]
        similarity = cosine_similarity(embeddings, embeddings)
        labels_sorted=labels.copy()
        labels_sorted.sort()
        similarity_df = pd.DataFrame(similarity, index=labels, columns=labels).sort_index()[labels_sorted]
        sns.heatmap(similarity_df, cmap=my_cmap)
        plt.show()
        
    def plot_embeddings(self, data_loader, batch_count, problem_index_to_id):
        
 
        embeddings, labels = self.get_embeddings_from_dls(data_loader, batch_count)
        print(embeddings.shape)
        print(labels)
        self.show_cosine_similarity(embeddings,labels,problem_index_to_id,100)
        if problem_index_to_id is not None:
            labels = [problem_index_to_id[yy] for yy in labels]
        
        tsne=sklearn.manifold.TSNE(n_components=2)
        batch_embeddings_2d=pd.DataFrame(tsne.fit_transform(embeddings) , columns=['x','y'])
        batch_embeddings_2d['label']=labels
        plt.figure(figsize=(10,10)) 
        sns.scatterplot(batch_embeddings_2d, x='x',y='y', hue='label', style='label',  palette=sns.color_palette(cc.glasbey, n_colors=24))
        plt.show()
    
            
    def shuffle_x_y(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def preprocess(self,sample_df):
        x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test = self.data_processor.run(sample_df, self.train_seeds, self.val_seeds)
        
        x_train, y_train = self.shuffle_x_y(x_train, y_train)
        x_val,y_val = self.shuffle_x_y(x_val,y_val)
        
        return x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id
    
    def save_embeddings(self, sample_df):
        x,y=self.data_processor.get_x_y(sample_df, shuffle=False)
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
            
        x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test = self.data_processor.run(sample_df, self.train_seeds, self.val_seeds)
        dset_train, dset_val, dset_test = [TSDatasets(np.swapaxes(xx,1,2),yy) for xx,yy in [(x_train,y_train),(x_val,y_val),(x_test,y_test)]]
        
        dls = TSDataLoaders.from_dsets(dset_train, dset_val, bs=self.batch_size)
        
        test_data_loader = TSDataLoaders.from_dsets(dset_test, bs=self.batch_size)[0]
        
        dls.c=len(set(y_train))
        print(f'Number of samples: {dls.len}, Number of variables: {dls.vars}, Number of classes: {dls.c}')
        self.model=OptTransStats(dls.vars, dls.c, dls.len, n_heads=self.n_heads, n_layers=self.n_layers, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, use_positional_encoding=self.use_positional_encoding, iteration_count=self.iteration_count, aggregations=self.aggregations)

        if plot_embeddings:
            self.plot_embeddings(test_data_loader, 100, problem_index_to_id )
        self.train_model(dls)
        self.evaluate(x_train,y_train,'train', problem_index_to_id, ids_train )
        self.evaluate(x_val,y_val,'val', problem_index_to_id, ids_val)
        test_report, test_confusion_matrix = self.evaluate(x_test,y_test,'test', problem_index_to_id, ids_test )
        
        if plot_embeddings:
            self.plot_embeddings(test_data_loader, 100, problem_index_to_id )
        
        embedding_df=None
        if save_embeddings:
            embedding_df=self.save_embeddings(sample_df)
        #self.check_random_forest_performance(dset_train, dset_val, dset_test)
        
        torch.save(self.model, os.path.join(self.result_dir,f'trained_model.pt'))
        return self.model, self.learner, test_report, test_confusion_matrix, embedding_df