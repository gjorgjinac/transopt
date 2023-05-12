from tsai_gina import *
from tsai.all import *
computer_setup()

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import sklearn 
from fastai.callback.tracker import EarlyStoppingCallback
import seaborn as sns
from clearml import Task

   
class GinaTST(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int, 
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,  
                 d_ff:int=256, dropout:float=0.1, act:str="gelu", fc_dropout:float=0., 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len
        
        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        #W_pos = torch.empty((q_len, d_model), device=default_device())
        #nn.init.uniform_(W_pos, -0.02, 0.02)
        #self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers)
        self.flatten = Flatten()
        
        # Head
        self.head_nf = q_len * d_model
        '''self.head =  torch.nn.Sequential(
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(self.head_nf, c_out, bias=True),
        )'''
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None, **kwargs):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)    
    
        
    def get_embeddings(self, x:Tensor):

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        #u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        #z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]
        return z

    def forward(self, x:Tensor, mask:Optional[Tensor]=None) -> Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        #u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)                                             # output: [bs x c_out]
    
    
def get_x_y(sample_df, verbose=True):
    if verbose:
        print('Extracting x and y')
    xys=[]
    problem_ids = np.unique(sample_df.index.get_level_values('problem_id').values)
    instance_ids = np.unique(sample_df.index.get_level_values('instance_id').values)
    for problem_id in problem_ids:
        problem_samples=sample_df.query("problem_id==@problem_id")
        for instance_id in instance_ids:
            problem_instance_samples=problem_samples.query("instance_id==@instance_id")
            xys+=[(problem_instance_samples.values,problem_id-1)]
        #print(problem_id)
    random.shuffle(xys)
    x=np.array([xy[0] for xy in xys])
    y=np.array([xy[1] for xy in xys])
    return x,y



def split_data(sample_df, verbose=True):
    if verbose:
        print('Splitting data')
    instance_ids = list(set(sample_df.index.get_level_values('instance_id').values))
    train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.1)
    train_instance_ids, val_instance_ids = train_test_split(train_instance_ids, test_size=0.1)

    train, val, test =[sample_df.query(f"instance_id in @split_instance_ids").sample(frac=1) for split_instance_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]
    return train, val, test



def preprocessing(split,split_name,scaler,dimension_reducer,verbose=True):
    if verbose:
        print('Doing preprocessing')
    split_og_index=split.copy().index
    split_og_columns = split.copy().columns
    if split_name=='train':
        scaler.fit(split)
        dimension_reducer.fit(split[x_columns])
    if normalize:
        split = scaler.transform(split)
        split = pd.DataFrame(split, index=split_og_index, columns=split_og_columns)
    if reduce:
        split_x = dimension_reducer.transform(split[x_columns])
        split=np.concatenate((split_x, split[y_columns]),axis=1)
    split = pd.DataFrame(split, index=split_og_index)
    return split, scaler, dimension_reducer


def scale_splits(train, val, test, scaler=None, dimension_reducer=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0,1))
    if scaler is None:
        dimension_reducer = PCA(1)
    train, scaler, dimension_reducer = preprocessing(train,'train',scaler,dimension_reducer)
    val, scaler, dimension_reducer = preprocessing(val,'val',scaler,dimension_reducer)
    test, scaler, dimension_reducer = preprocessing(test,'test',scaler,dimension_reducer)
    return train, val, test


def find_learning_rate(dls):
    print('Determining learning rate')
    model = TST(dls.vars, dls.c, dls.len)
    learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAuc(), accuracy],  cbs=ShowGraphCallback2())
    learn.lr_find()
    
    
def train_model(model,dls, plot_training=True):
    print('Training model')
    callbacks=[EarlyStoppingCallback( min_delta=0.001,patience=10)]
    #callbacks=[]
    if plot_training:
        callbacks+=[ShowGraphCallback2()]
    
    learner = Learner(dls, model, loss_func=CrossEntropyLoss(), metrics=[ accuracy],  cbs=callbacks)
    start = time.time()
    learner.fit_one_cycle(100, lr_max=5e-4)
    #learner.fit(20, lr=1e-5)
    print('\nElapsed time:', time.time() - start)
    if plot_training:
        learner.plot_metrics()
    return model,learner

def evaluate(learner, x_test, y_test):
    print('Evaluating model')
    test_probas, test_targets, test_preds = learner.get_X_preds(np.swapaxes(x_test,1,2), with_decoded=True)
    test_predicted_classes = [int(t.argmax()) for t in test_probas]
    print(pd.DataFrame(classification_report(y_test, test_predicted_classes, output_dict=True)))
    #print(pd.DataFrame(confusion_matrix(y_test, test_predicted_classes)))
    ConfusionMatrixDisplay.from_predictions(y_test, test_predicted_classes)

    
def get_batch_embeddings(model,batch):
    batch_embeddings=model.cuda().get_embeddings(batch[0].cuda())
    batch_embeddings=batch_embeddings.detach().cpu().numpy()
    np.swapaxes(batch_embeddings,1,2)
    return batch_embeddings


def get_embeddings_from_dls(model,dls, batch_count=100):
    batch = dls.one_batch()
    all_embeddings=get_batch_embeddings(model,batch)
    all_labels=list(batch[1])
    i=0
    while batch is not None and i < batch_count:
        batch = dls.one_batch()
        all_embeddings=np.append(all_embeddings,get_batch_embeddings(model,batch), axis=0)
        all_labels+=list(batch[1])
        i+=1
    all_labels=[int(i) for i in all_labels]
    return all_embeddings, all_labels


def plot_embeddings(model,dls, batch_count):
    embeddings, labels = get_embeddings_from_dls(model,dls, batch_count)
    tsne=sklearn.manifold.TSNE(n_components=2)
    batch_embeddings_2d=pd.DataFrame(tsne.fit_transform(embeddings.mean(axis=2)) , columns=['x','y'])
    batch_embeddings_2d['label']=labels
    plt.figure(figsize=(10,10)) 
    sns.scatterplot(batch_embeddings_2d, x='x',y='y', hue='label', style='label')
    plt.show()
    
    
def run (sample_df, plot_training=True, verbose=True, n_heads=1, n_layers=1, d_model=20, d_k=10, d_v=10, n_epochs=100, bs=8):
    train, val, test = split_data(sample_df,verbose=verbose)
    x_train, y_train= get_x_y(train,verbose=verbose)
    x_val,y_val=get_x_y(val,verbose=verbose)
    x_test, y_test= get_x_y(test,verbose=verbose)

    dset_train, dset_val, dset_test = [TSDatasets(np.swapaxes(xx,1,2),yy) for xx,yy in [(x_train,y_train),(x_val,y_val),(x_test,y_test)]]
    
    dls = TSDataLoaders.from_dsets(dset_train, dset_val, bs=bs)
    dls.c=len(set(y_train))
    print(f'Number of samples: {dls.len}, Number of variables: {dls.vars}, Number of classes: {dls.c}')
    model = GinaTST(dls.vars, dls.c, dls.len, n_heads=n_heads, n_layers=n_layers, d_model=d_model, d_k=d_k, d_v=d_v)
    plot_embeddings(model,dls, 200)
    model,learner = train_model(model,dls, plot_training=plot_training)
    evaluate(learner,x_train,y_train)
    evaluate(learner,x_val,y_val)
    evaluate(learner,x_test,y_test)
    
    plot_embeddings(model,dls, 200)
    return model, learner, train, val, test