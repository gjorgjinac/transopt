from tsai_custom import *
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
   
class OptTransStats(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int,
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,  
                 d_ff:int=256, dropout:float=0.1, act:str="relu", fc_dropout:float=0, use_positional_encoding=False,
                 y_range:Optional[tuple]=None, verbose:bool=False, aggregations=None, do_regression=False, **kwargs):
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
        self.use_positional_encoding=use_positional_encoding
        self.aggregations=aggregations if aggregations is not None else ['min','max','std','mean']
        # Positional encoding
        

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers)
        
        # Head
        self.head_nf = d_model * len(self.aggregations)
        self.head=torch.nn.Sequential(
               torch.nn.Linear(self.head_nf, 100, bias=True),
              torch.nn.ReLU(),
            
            Flatten(),
            nn.Dropout(dropout),
            torch.nn.Linear(100, c_out, bias=True),
        )
        
    #self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None, **kwargs):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)    
    
    def get_stats(self,z):
        
        z= z.type(torch.float)
        z_min = torch.min(z, 1).values.cuda()
        z_max = torch.max(z, 1).values.cuda()
        z_mean = torch.mean(z,1, dtype=float).cuda()
        z_std = torch.std(z,1, unbiased=False).cuda()
        aggregated_values={'min':z_min,'max':z_max, 'mean':z_mean, 'std':z_std}
        try:
            if self.aggregations is None:
                self.aggregations=aggregated_values.keys()
        except Exception:
            self.aggregations=aggregated_values.keys()
        z = torch.cat([aggregated_values[a] for a in self.aggregations],1)
        z= z.type(torch.float)
        return z
    def get_embeddings(self, x:Tensor):

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        if self.use_positional_encoding:
            u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        #z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]
        z = self.get_stats(z)
        return z

    def forward(self, x:Tensor, mask:Optional[Tensor]=None) -> Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]
    
        if self.use_positional_encoding:
            u = self.dropout(u + self.W_pos)
        
        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        #z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]
        z = self.get_stats(z)
        # Classification/ Regression head
        return self.head(z)                                             # output: [bs x c_out]