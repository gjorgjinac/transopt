import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
import sys
from tsai_gina import *
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.preprocessing import MinMaxScaler

from pytorchtools import EarlyStopping

import torch
from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len


def preprocess_ela(train,val,test):
    train=train.select_dtypes(exclude=['object'])
    
    train=train.replace([np.inf, -np.inf], np.nan)
    test=test.replace([np.inf, -np.inf], np.nan)
    val=val.replace([np.inf, -np.inf], np.nan)
    count_missing={column:train[column].isna().sum() for column in train.columns}
    count_missing=pd.DataFrame([count_missing]).T
    count_missing.columns=['missing']
    count_missing['missing_percent']=count_missing['missing'].apply(lambda x: x/train.shape[0])
    
    columns_to_keep=list(count_missing.query('missing_percent<0.1').index)
    print('Keeping columns', columns_to_keep)
    train=train[columns_to_keep]
    test=test[columns_to_keep]
    val=val[columns_to_keep]
    
    train = train.fillna(train.mean()).astype(np.float32)
    val = val.fillna(train.mean()).astype(np.float32)
    test = test.fillna(train.mean()).astype(np.float32)
    minmax=MinMaxScaler()
    cols=train.columns
    train=pd.DataFrame( minmax.fit_transform(train), columns = cols)
    test=pd.DataFrame( minmax.transform(test), columns = cols)
    val=pd.DataFrame( minmax.transform(val), columns = cols)
    return train,val,test

def train_model(model, train_loader, valid_loader,batch_size, patience, n_epochs, min_delta, checkpoint_file):
    best_model=None
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=min_delta, path=checkpoint_file)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_file))

    return  model, avg_train_losses, avg_valid_losses


dimension=int(sys.argv[1])
group=sys.argv[2]
checkpoint_file=f'checkpoints/checkpoint_dim_{dimension}_group_{group}.pt'
ela=pd.read_csv(f'ela_data/lhs_samples_dimension_{dimension}_50_samples.csv',index_col=[0])
ela['problem_id']=ela['problem_id'].apply(lambda x: x-1)
ela=ela.set_index(['problem_id','instance_id'])
result_dir='results_ela_nn'
ela_groups=list(set([x.split('.')[0] for x in ela.columns])) + ['all']

print(ela_groups)
all_results=[]

for fold in range(0,10):
    run_name=f'dim_{dimension}_fold_{fold}_features_{group}'
    if os.path.isfile(f'{result_dir}/{run_name}_classification_report.csv'):
        continue
    train_instance_ids, val_instance_ids, test_instance_ids = [list(pd.read_csv(f'folds/problem_classification_999_instances/{split_name}_{fold}.csv',index_col=[0])['0'].values) for split_name in ['train','val','test']]
    if group=='all':
        group_features=ela.columns
    else:
        group_features=list(filter(lambda c: c.startswith(f'{group}.'), ela.columns))

    x_train, x_val, x_test= [ela.query('instance_id in @split_ids')[group_features] for split_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]

    y_train, y_val, y_test = [d.reset_index()['problem_id'] for d in [x_train, x_val, x_test]]

    x_train,x_val,x_test=preprocess_ela(x_train,x_val,x_test)

    print(x_train.shape)
    print(y_train.shape)
    #x_train, y_train, x_val, y_val, x_test, y_test, problem_id_to_index, problem_index_to_id, ids_train, ids_val, ids_test = self.data_processor.run(sample_df, self.train_seeds, self.val_seeds)
    #dset_train, dset_val, dset_test = [TorchDataset (xx,yy) for xx,yy in [(x_train,y_train),(x_val,y_val),(x_test,y_test)]]

    traindata = Data(x_train.values,y_train.values)
    valdata = Data(x_val.values,y_val.values)
    testdata = Data(x_test.values,y_test.values)



    trainloader = DataLoader(traindata, batch_size=8, 
                     shuffle=True, num_workers=2)
    valloader = DataLoader(valdata, batch_size=8, 
                     shuffle=True, num_workers=2)
    testloader = DataLoader(testdata, batch_size=8, 
                     shuffle=True, num_workers=2)

    model=torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 120, bias=True),
        torch.nn.Linear(120, 100, bias=True),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(100, 24, bias=True),
    )

    epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, avg_train_losses, avg_valid_losses = train_model(model, trainloader, valloader, 8, 5, 200, 0.001, checkpoint_file=checkpoint_file)
    test_preds=[]
    test_ys=[]
    for batch, (data, target) in enumerate(testloader, 1):
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        preds=[np.argmax(o.detach().numpy() ) for o in output]
        test_preds+=preds
        test_ys+=target
    print('Accuracy')
    test_accuracy=accuracy_score(test_ys,test_preds)
    print(test_accuracy)
    all_results+=[(group, fold,test_accuracy)]
    pd.DataFrame(all_results, columns = ['group','fold','accuracy']).to_csv(f'results_ela_nn/dimension_{dimension}_group_{group}.csv')
pd.DataFrame(all_results, columns = ['group','fold','accuracy']).to_csv(f'results_ela_nn/dimension_{dimension}_group_{group}.csv')